"""
    FixedCoordinates(problem, free, base)

Conditional log density in the coordinates indexed by `free`, in that order.
Every other coordinate retains its value in the full-length vector `base`.
"""
struct FixedCoordinates{P}
    problem::P
    free::Vector{Int}
    base::Vector{Float64}
end

LogDensityProblems.capabilities(::Type{<:FixedCoordinates}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(f::FixedCoordinates) = length(f.free)
_own_gradient_backend(f::FixedCoordinates) = _own_gradient_backend(f.problem)

function _full_position(f::FixedCoordinates, x)
    y = copy(f.base)
    y[f.free] = x
    y
end
LogDensityProblems.logdensity(f::FixedCoordinates, x) =
    LogDensityProblems.logdensity(f.problem, _full_position(f, x))
function LogDensityProblems.logdensity_and_gradient(f::FixedCoordinates, x)
    value, gradient = LogDensityProblems.logdensity_and_gradient(f.problem, _full_position(f, x))
    value, gradient[f.free]
end

struct CoopFamily{T,I}
    targets::T
    ids::I
    features::Union{Nothing,Matrix{Float64}}
    # Zero denotes a first-wave slot. Other slots store the index of their medoid.
    source::Vector{Int}
    partners::Vector{Vector{Int}}
    pilot_partners::Vector{Vector{Int}}
end
_coop_target(family::CoopFamily, i) = family.targets[i]

struct CoopFamilyInit{I}
    starts::I
end
function _coop_cold_init(init::CoopFamilyInit, i, rng)
    starts = init.starts
    starts isa AbstractVector && return starts[i]
    applicable(starts, rng, i) ? starts(rng, i) : starts(i)
end

function _standardize_features(features, m)
    matrix = if features isa AbstractMatrix
        Matrix{Float64}(features)
    elseif features isa AbstractVector{<:Real}
        reshape(collect(Float64, features), 1, :)
    else
        length(features) == m || throw(DimensionMismatch("expected one feature vector per target"))
        Matrix{Float64}(stack(features))
    end
    size(matrix, 2) == m && size(matrix, 1) > 0 ||
        throw(DimensionMismatch("features must have one column per target and at least one row"))
    all(isfinite, matrix) || throw(ArgumentError("target features must be finite"))
    (matrix .- mean(matrix; dims=2)) ./ max.(std(matrix; dims=2, corrected=false), 1e-12)
end

# Greedy medoid BUILD on a bounded representative sample. Its working set
# depends on the wave size; the full family only needs distances to its medoids.
function _wave_sources(features, k)
    m = size(features, 2)
    k == m && return zeros(Int, m)
    rng = Random.Xoshiro(0)
    sample_size = min(m, max(k, 128))
    indices = sort!(StatsBase.sample(rng, 1:m, sample_size; replace=false))
    medoids = if k == sample_size
        indices
    else
        # BUILD needs the gain in coverage, not a distance matrix. Even for
        # a small family we only keep one distance per candidate.
        sample = view(features, :, indices)
        distance(i, j) = sqrt(sum(abs2(sample[a,i] - sample[a,j]) for a in axes(sample, 1)))
        j = argmin([sum(distance(i, j) for i in 1:sample_size) for j in 1:sample_size])
        selected = falses(sample_size)
        selected[j] = true
        closest = [distance(i, j) for i in 1:sample_size]
        for _ in 2:k
            gains = [selected[j] ? -Inf : sum(max(closest[i] - distance(i, j), 0.0)
                     for i in 1:sample_size) for j in 1:sample_size]
            j = argmax(gains)
            selected[j] = true
            for i in 1:sample_size
                closest[i] = min(closest[i], distance(i, j))
            end
        end
        indices[selected]
    end
    tree = NearestNeighbors.KDTree(features[:, medoids])
    sources = [medoids[first(NearestNeighbors.nn(tree, view(features, :, i)))] for i in 1:m]
    sources[medoids] .= 0
    sources
end

function CoopFamily(targets, ids, features, n_wave1)
    m = length(targets)
    m > 0 || throw(ArgumentError("a family must contain at least one target"))
    d = LogDensityProblems.dimension(first(targets))
    d > 0 && all(t -> LogDensityProblems.dimension(t) == d, targets) ||
        throw(DimensionMismatch("all family targets need the same positive dimension"))
    ids === nothing || (length(ids) == m && allunique(ids)) ||
        throw(ArgumentError("target_ids must identify every target uniquely"))
    n_wave1 >= 1 || throw(ArgumentError("n_wave1 must be positive"))
    F = isnothing(features) ? nothing : _standardize_features(features, m)
    source = isnothing(F) ? zeros(Int, m) : _wave_sources(F, min(n_wave1, m))
    partners = if isnothing(F)
        [[mod1(i + k, m) for k in 1:min(8, m-1)] for i in 1:m]
    else
        _feature_neighbors(F, 1:m)
    end
    pilot_partners = any(!iszero, source) ? _feature_neighbors(F, findall(iszero, source)) : partners
    CoopFamily(targets, ids, F, source, partners, pilot_partners)
end

# Query a bounded number of neighbors, once. During the first wave many of
# the nearest targets in the full family have yet to run a single transition.
function _feature_neighbors(features, indices)
    tree = NearestNeighbors.KDTree(features[:, indices])
    map(axes(features, 2)) do i
        js, _ = NearestNeighbors.knn(tree, view(features, :, i), min(9, length(indices)), true)
        neighbors = filter(!=(i), indices[js])
        first(neighbors, min(8, length(neighbors)))
    end
end

# First-wave draws used to estimate the contraction of a displaced start.
# Keep this short history separate from the consumer's requested draws.
const _COOP_PILOT_DRAWS = 30
mutable struct CoopFreeze
    warmup::Int
    wave::Int
    source::Int
    pilot_quota::Int
    pilot::Vector{Vector{Float64}}
    extensions::Int              # current qualification attempt
    total_extensions::Int
    prior_rescues::Int
    scores::Vector{Float64}
    abandoned_scores::Vector{Float64}
    pending::Bool
    clean::Bool
    warmup_transitions::Int
    divergences::Int
end
CoopFreeze(warmup) = CoopFreeze(warmup, 1, 0, 0, Vector{Float64}[], 0, 0, 0,
    Float64[], Float64[], false, false, 0, 0)

function _second_wave_length(donors)
    length(donors) >= 2 || return 50
    segments = map(donors) do c
        draws = stack(c.freeze.pilot)
        reparametrize!(c.lpdf, draws)
        draws
    end
    means = [vec(mean(x; dims=2)) for x in segments]
    spread = sqrt.(max.(mean(vec(var(x; dims=2)) for x in segments), 1e-12))
    centered = permutedims(stack([x .- mu for (x, mu) in zip(segments, means)]), (2,3,1))
    ess = MCMCDiagnosticTools.ess(centered; kind=:basic)
    all(isfinite, ess) || return 50
    tau = size(centered,1) * size(centered,2) ./ max.(ess, 1)
    rho = clamp.((tau .- 1) ./ (tau .+ 1), 0, 0.99)
    standardized = stack(means) ./ spread
    tree = NearestNeighbors.KDTree(standardized)
    offsets = map(eachindex(donors)) do i
        js, _ = NearestNeighbors.knn(tree, view(standardized, :, i), 2, true)
        j = first(filter(!=(i), js))
        abs.((means[i] .- means[j]) ./ spread)
    end
    delta = vec(median(stack(offsets); dims=2))
    rms = median([sqrt(mean(abs2, o)) for o in offsets])
    energy = rms <= 0.1 ? 0.0 : 2log(rms / 0.1) / -log(0.5)
    mixing = maximum(d <= 0.1 || r <= 0 ? 0.0 : log(d/0.1) / -log(r) for (d,r) in zip(delta,rho))
    ceil(Int, clamp(max(energy, mixing), 2, 50))
end

"""
    cooperative_warmup_mcmc(rng, targets::AbstractVector; init, n_draws,
        features=nothing, n_wave1=16, target_ids=nothing, kwargs...)

Run one chain per target, in input order. Supply one initializer per target,
or a function `init(rng, i)` or `init(i)` producing its cold initializer.
A vector of feature vectors (or a feature-by-target matrix) enables a first
wave of medoids. Each remaining chain inherits its nearest medoid's state,
metric and step size and warms up on its own target before keeping a draw.
With no features, every chain starts cold. Family targets must be thread-safe.

Only within-target variation trains a family metric. Each target retains its
own reparametrization controls. The result reports each slot's wave, source
(zero for a cold start), warm-up transitions, divergences and freeze extensions
in `results[i].accounting`. The gradient evaluation count follows the sampler's
evaluation budget, including receiving-target evaluations and discarded
transitions. Initializer and initial step-size searches are excluded. A cold
replacement preserves the earlier counts and exposes its predecessor's gate
scores as `abandoned_freeze_scores`. A budget can leave a slot `:waiting`.
Family `joint_ess` is `nothing` and `n_pooled` is zero. Use `target_ids` to
identify the ordered targets when checkpointing and supply the same IDs,
features and wave size on resume.
"""
function cooperative_warmup_mcmc(rng::Random.AbstractRNG, targets::AbstractVector;
        init, features=nothing, n_wave1::Integer=16, target_ids=nothing,
        n_chains::Integer=length(targets), n_draws::Integer=typemax(Int), target_ess=Inf, checkpoint_dir=nothing,
        nonlinear_adapt=false, kwargs...)
    n_chains == length(targets) || throw(DimensionMismatch("a family needs one chain per target"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    target_ess == Inf || throw(ArgumentError("target_ess cannot pool distinct targets; use n_draws or a budget"))
    nonlinear_adapt && throw(ArgumentError("family targets use their own fixed reparametrizations"))
    isnothing(checkpoint_dir) || !isnothing(target_ids) ||
        throw(ArgumentError("supply ordered target_ids when checkpointing a family"))
    if init isa AbstractVector
        length(init) == n_chains || throw(DimensionMismatch("init needs one entry per target"))
    elseif !applicable(init, rng, 1) && !applicable(init, 1)
        throw(ArgumentError("supply one init per target or init(rng, i) or init(i)"))
    end
    family = CoopFamily(targets, target_ids, features, n_wave1)
    cooperative_warmup_mcmc(rng, family; init=CoopFamilyInit(init), n_chains,
        n_draws, target_ess, checkpoint_dir, nonlinear_adapt, kwargs...)
end

_family_checkpoint(lpdf) = nothing
_family_checkpoint(f::CoopFamily) = (; target_ids=f.ids, features=f.features, source=f.source)

function _qualify_freezes!(chains)
    scores = Float64[]
    for c in chains, score in c.freeze.scores
        isfinite(score) && push!(scores, score)
    end
    limit = isempty(scores) ? 10.0 : max(10.0, 3median!(scores))
    for chain in chains
        f = chain.freeze
        f.pending || continue
        f.pending = false
        score = last(f.scores)
        if f.clean && isfinite(score) && score <= limit
            chain.status = :sampling
            chain.sampling_start = length(chain.acceptances)
            chain.stepsize = last(chain.stepsizes)   # the kernel just checked
        elseif f.extensions < 30
            f.extensions += 1
            f.total_extensions += 1
        else
            chain.status = :needs_replacement
        end
    end
end

_coop_rescue_count(batches) = count(b -> get(b, :recovery, :none) === :rescue, batches)

function _coop_carry_freeze!(new, old)
    # A replacement qualifies on fresh evidence. Keep the discarded
    # generation's checks available for accounting outside the active median.
    append!(new.freeze.abandoned_scores, old.freeze.abandoned_scores)
    append!(new.freeze.abandoned_scores, old.freeze.scores)
    new.freeze.total_extensions = old.freeze.total_extensions
    new.freeze.prior_rescues = old.freeze.prior_rescues + _coop_rescue_count(old.batches)
    new.freeze.warmup_transitions += old.freeze.warmup_transitions
    new.freeze.divergences += old.freeze.divergences
    new
end

function _coop_replacement_slot!(new, old, family::CoopFamily)
    _coop_carry_freeze!(new, old)
    new.freeze.wave = old.freeze.wave
    new.freeze.source = old.freeze.source
    new.freeze.pilot_quota = old.freeze.pilot_quota
    new
end

function _start_second_wave!(run)
    waiting = filter(c -> c.status === :waiting, run.chains)
    isempty(waiting) && return
    donors = filter(c -> c.freeze.wave == 1, run.chains)
    all(c -> length(c.freeze.pilot) == _COOP_PILOT_DRAWS, donors) || return
    warmup = _second_wave_length(donors)
    _parallel_foreach(waiting, run.cfg.n_cores) do chain
        _stop_requested(run) && return
        source = run.chains[chain.freeze.source]
        # A new conditional always evaluates its own gradient and density.
        old_ir, new_ir = reparametrizer(source.lpdf), reparametrizer(chain.lpdf)
        x = copy(source.position_and_gradient.q)
        if has_reparametrization(old_ir) || has_reparametrization(new_ir)
            _transport_position!(x, similar(x), new_ir, old_ir, source.position_and_gradient.q)
        end
        point = _evaluate_if_finite(chain.lpdf, x)
        chain.total_evaluation_counter += 1
        Threads.atomic_add!(run.total_evaluations, 1)
        if isnothing(point)
            chain.status = :needs_replacement
        else
            chain.position_and_gradient = point
            chain.scale = deepcopy(source.scale)
            chain.kinetic_energy = _coop_energy(chain.scale)
            chain.log_kappa = chain.kappa_prior = source.log_kappa
            chain.stepsize = source.stepsize
            chain.freeze.warmup = warmup
            chain.status = :warming
        end
    end
end

# Family recovery uses the agreed geometry among the receiver's feature
# neighbors. Positions and log densities never enter a between-target test.
struct CoopFamilyRecovery{C,R}
    chains::C
    responsibilities::R
    membership::Vector{Int}
    partners::Vector{Vector{Int}}
end

function _coop_recovery_context(run, family::CoopFamily)
    R = run.clusters.responsibilities
    membership = [_coop_geometry_current(c) && c.current_transition_counter > 10 ?
                  argmax(view(R, c.chain_index, :)) : 0 for c in run.chains]
    partners = any(c -> c.status === :waiting, run.chains) ? family.pilot_partners : family.partners
    CoopFamilyRecovery(run.chains, R, membership, partners)
end

function _family_donors(context, chain)
    candidates = filter(i -> context.membership[i] > 0, context.partners[chain.chain_index])
    length(candidates) >= 2 || return nothing
    groups = context.membership[candidates]
    majority = StatsBase.mode(groups)
    count(==(majority), groups) > length(groups) ÷ 2 || return nothing
    step = median([context.chains[i].stepsize for i in candidates])
    donors = filter(candidates) do i
        c = context.chains[i]
        context.membership[i] == majority && c.stepsize >= step / 10 &&
            _coop_divergent_fraction(c) <= 0.4 && c.moments.weight >= 2
    end
    length(donors) >= 2 || return nothing
    (; majority, donors, step, isolated=context.membership[chain.chain_index] ∉ groups)
end

function _coop_recovery_action(context::CoopFamilyRecovery, chain, batch)
    context.membership[chain.chain_index] > 0 || return _NO_COOP_RECOVERY
    group = _family_donors(context, chain)
    isnothing(group) && return _NO_COOP_RECOVERY
    efficiencies = filter(isfinite, [_coop_ess_per_draw(context.chains[i]) for i in group.donors])
    efficiency = isempty(efficiencies) ? NaN : median(efficiencies)
    _coop_recovery_action(chain, batch, group.step, efficiency, group.isolated)
end

function _coop_rescue_geometry(context::CoopFamilyRecovery, chain)
    group = _family_donors(context, chain)
    donors = view(context.chains, group.donors)
    weights = view(context.responsibilities, group.donors, group.majority)
    moments = _combine_moments!(CoopMoments(chain.dimension), (c.moments for c in donors),
                                weights; within_target=true)
    log_kappa = sum(w * c.log_kappa for (c, w) in zip(donors, weights)) / sum(weights)
    (; moments, log_kappa, cluster=group.majority, group.donors)
end
