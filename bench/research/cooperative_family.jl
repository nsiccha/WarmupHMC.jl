# Reproduce with the package test environment. Modes: family [M] [seed],
# shared [seed], or memory. The covariance below belongs to the synthetic
# target; every sampler starts with the same unit diagonal scale.
using WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems
using Distributions, MCMCDiagnosticTools
BLAS.set_num_threads(1)

struct GaussianConditional
    mu::Vector{Float64}
    factor::LowerTriangular{Float64,Matrix{Float64}}
    gradients::Threads.Atomic{Int}
end

# All arrays are read-only, including when a target is shared across chains.
# Count calls to the actual gradient, including initialization and barrier checks.
Base.deepcopy_internal(t::GaussianConditional, ::IdDict) = t
LogDensityProblems.dimension(t::GaussianConditional) = length(t.mu)
LogDensityProblems.capabilities(::Type{GaussianConditional}) = LogDensityProblems.LogDensityOrder{1}()
residual(t, x) = t.factor \ (x - t.mu)
LogDensityProblems.logdensity(t::GaussianConditional, x) = -sum(abs2, residual(t, x)) / 2
function LogDensityProblems.logdensity_and_gradient(t::GaussianConditional, x)
    Threads.atomic_add!(t.gradients, 1)
    z = residual(t, x)
    -sum(abs2, z)/2, -(t.factor' \ z)
end

function gaussian_family(m, rng; d=50)
    F = randn(rng, 3, m)
    rotation = Matrix(qr(randn(rng, d, d)).Q)
    baseline = rotation * Diagonal(10.0 .^ range(0,-1; length=d))
    location = 2baseline * randn(rng, d, 3) / sqrt(3)
    shape = [Symmetric((A+A')/(2sqrt(d))) for A in (randn(rng,d,d) for _ in 1:3)]
    targets = map(eachcol(F)) do u
        # Exponential perturbation keeps all conditionals nonsingular.
        L = baseline * exp(Symmetric(0.2sum(A*w for (A,w) in zip(shape,u))))
        GaussianConditional(location * u, cholesky(Symmetric(L * L')).L, Threads.Atomic{Int}(0))
    end
    targets, F
end
cold_init(rng, d) = (; position=3randn(rng,d), squared_scale=ones(d))

function calibration(targets, draws, source)
    m = length(draws)
    d = length(first(draws))
    Z = stack(residual(t,x) for (t,x) in zip(targets,draws))
    marginal = stack((x - t.mu) ./ sqrt.(vec(sum(abs2, t.factor; dims=2))) for (t,x) in zip(targets,draws))
    radial_pit = [cdf(Chisq(d), sum(abs2,z)) for z in eachcol(Z)]
    # A group contributes all pairs, using the sum of coordinates and their
    # squared norms so reporting stays linear in the number of draws.
    groups = Dict{Int,Vector{Int}}()
    for (i,j) in enumerate(source)
        j > 0 && push!(get!(groups, j, Int[]), i)
    end
    pairs = 0
    coord_cross = coord_sum = coord_sq = 0.0
    norm_cross = norm_sum = norm_sq = 0.0
    for indices in values(groups)
        group = view(Z, :, indices)
        pairs += length(indices) * (length(indices)-1) ÷ 2
        coord_cross += sum(abs2, sum(group; dims=2)) - sum(abs2, group)
        coord_sum += (length(indices)-1) * sum(group)
        coord_sq += (length(indices)-1) * sum(abs2, group)
        radial = vec(sum(abs2, group; dims=1))
        norm_cross += sum(radial)^2 - sum(abs2,radial)
        norm_sum += (length(indices)-1) * sum(radial)
        norm_sq += (length(indices)-1) * sum(abs2,radial)
    end
    # Each end of an unordered pair appears once on each side. These are
    # Pearson correlations of all such ordered pairs, without storing pairs.
    function pair_correlation(cross, total, squares, n)
        n == 0 && return nothing
        correction = total^2 / n
        squares > correction ? (cross - correction) / (squares - correction) : nothing
    end
    coord_cor = pair_correlation(coord_cross, coord_sum, coord_sq, 2pairs*d)
    norm_cor = pair_correlation(norm_cross, norm_sum, norm_sq, 2pairs)
    summarize(p) = (; mean=mean(p), variance=var(p), tails=mean(x -> x < 0.025 || x > 0.975, p))
    (; targets=m, marginal_pit=summarize(cdf.(Normal(), marginal)),
       radial_pit=summarize(radial_pit), pairs, shared_source=coord_cor, shared_squared_norm=norm_cor,
       far=count(z -> maximum(abs,z) > 6, eachcol(Z)),
       far_marginal=count(z -> maximum(abs,z) > 6, eachcol(marginal)), max_whitened=maximum(abs,Z))
end

function family_check(m, seed)
    targets, features = gaussian_family(m, Xoshiro(seed))
    inits = [cold_init(Xoshiro(100seed+i), 50) for i in 1:m]
    family = cooperative_warmup_mcmc(Xoshiro(200seed), targets; init=inits, features, n_draws=1)
    family_gradients = sum(t.gradients[] for t in targets)
    println("family M=",m," stop=",family.stop_reason," used=",family.n_used,
        " transition_evaluations=",family.total_evaluation_counter, " gradients=",family_gradients,
        " extensions=",sum(r.accounting.freeze_extensions for r in family.results),
        " replacements=",sum(get(r,:generation,1) - 1 for r in family.results))
    indexes = findall(r -> r.n_samples == 1, family.results)
    # Replacement generations start cold. Only retained draws actually
    # initialized by the same first-wave chain share a donor.
    sources = [get(r, :generation, 1) == 1 ? r.accounting.source : 0 for r in family.results]
    missing_slots = [(; r.chain_index, r.status, r.accounting, generation=get(r,:generation,1))
                     for r in family.results if r.n_samples == 0]
    isempty(missing_slots) || println("missing target slots ", missing_slots)
    if !isempty(indexes)
        observed = calibration(targets[indexes], [vec(family.results[i].posterior_position) for i in indexes], sources[indexes])
        println("cooperative retained ", observed)
        @assert observed.far == observed.far_marginal == 0
    end
    # One independently seeded cold fit and one preselected retained draw per target.
    cold = Vector{Vector{Float64}}(undef,m)
    evals = zeros(Int,m)
    Threads.@threads for i in 1:m
        r = adaptive_warmup_mcmc(Xoshiro(900seed+i), targets[i]; init=inits[i], n_draws=100,
            blas_threads=nothing)
        cold[i] = r.posterior_position[:,mod1(i,100)]
        evals[i] = r.total_evaluation_counter
    end
    controls = calibration(targets,cold,sources)
    println("independent cold all targets ",controls,
        " transition_evaluations=",sum(evals)," gradients=",sum(t.gradients[] for t in targets)-family_gradients)
    @assert length(indexes) == m && family.stop_reason === :n_draws "incomplete target family"
    @assert controls.far == controls.far_marginal == 0
end

function shared_check(seed)
    targets, _ = gaussian_family(1, Xoshiro(seed))
    p = only(targets)
    inits = [cold_init(Xoshiro(100seed+i),50) for i in 1:16]
    shared = cooperative_warmup_mcmc(Xoshiro(300seed), p; init=inits, n_chains=16, n_draws=400)
    shared_gradients = p.gradients[]
    cold = adaptive_warmup_mcmc([Xoshiro(500seed+i) for i in 1:16], p; init=inits, n_draws=400)
    for (label, results, gradients) in (("shared", shared.results, shared_gradients),
            ("independent", cold, p.gradients[] - shared_gradients))
        A = permutedims(stack(Matrix(r.posterior_position) for r in results), (2,3,1))
        bulk = minimum(ess(A; kind=:bulk))
        n = sum(r.total_evaluation_counter for r in results)
        max_z = maximum(maximum(abs, residual(p,x)) for r in results for x in eachcol(r.posterior_position))
        marginal_sd = sqrt.(vec(sum(abs2, p.factor; dims=2)))
        max_marginal = maximum(maximum(abs, (x - p.mu) ./ marginal_sd) for r in results for x in eachcol(r.posterior_position))
        println(label," transition_evaluations=", n, " gradients=",gradients,
            " min_bulk_ess=", bulk, " gradients_per_ess=",gradients/bulk, " max_whitened=",max_z,
            " max_marginal=",max_marginal, " min_tail_ess=",minimum(ess(A;kind=:tail)), " max_rhat=",maximum(rhat(A)),
            " divergent_kept=",sum(r.n_divergent_samples for r in results))
        println(label, " first draw ", calibration(fill(p,16), [r.posterior_position[:,1] for r in results], zeros(Int,16)))
        @assert max(max_z, max_marginal) <= 6
    end
end

function memory_check()
    targets, features = gaussian_family(1, Xoshiro(1))
    target = only(targets)
    build(T,F) = WarmupHMC.CoopFamily(T, nothing, F, 16)
    initialize(T,F) = cooperative_warmup_mcmc(Xoshiro(33), T; features=F,
        init=i -> cold_init(Xoshiro(2000+i),50), n_cores=1, n_draws=1, n_evaluations_budget=0)
    build(fill(target,32), randn(Xoshiro(1),3,32))
    initialize(fill(target,32), randn(Xoshiro(1),3,32))
    for m in (256,512,1024,2048)
        T, F = fill(target,m), randn(Xoshiro(31),3,m)
        run = @timed build(T,F)
        initial = @timed initialize(T,F)
        @assert initial.value.n_used == 0
        println("M=",m," setup_allocation_bytes=",initial.bytes," feature_allocation_bytes=",run.bytes,
            " plan_bytes=",Base.summarysize(run.value),
            " partners=",sum(length,run.value.partners))
    end
end

if ARGS[1] == "family"
    family_check(parse(Int, ARGS[2]), parse(Int, get(ARGS,3,"1")))
elseif ARGS[1] == "shared"
    shared_check(parse(Int, get(ARGS,2,"1")))
elseif ARGS[1] == "memory"
    memory_check()
else
    error("expected family, shared, or memory")
end
