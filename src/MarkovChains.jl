# Latent Markov chains (random walks, AR(1) and OU series) whose states are
# centred adaptively per time step (snag random-walk-bloc-fc175e21, decision
# 1idvzrg). A chain is carried by an `IndexedReparametrization` next to its
# pairs and takes part in every path the pairs do: the map and its inverse, the
# gradient (one Enzyme reverse sweep per chain in `WarmupHMCEnzymeExt`), the
# window-end transport, selection, controls, snapshots and checkpoints.

"""
    MarkovChain(states; log_scale, coefficient = 1.0, drift = 0.0, initial = 0.0,
                phi = 0.0, c = 1.0)

A latent Markov chain, such as a random walk, an AR(1) series or a discretely
observed Ornstein–Uhlenbeck process, whose states warm-up centres adaptively,
with two controls per time step. Pass chains to an
[`IndexedReparametrization`](@ref) with `chains = [...]`.

`states` lists the chain's coordinates in time order, and the model is written in
the states `w_t` themselves. State `t`'s location `m_t` is `initial` for the
first state and `coefficient_t * w_{t-1} + drift_t` for every later one, with the
previous state read in the model's frame. The sampler works in

```
x_t = (w_t - phi_t * m_t) * exp((c_t - 1) * log_scale_t)
```

so `(phi_t, c_t) = (0, 1)` is the state itself (the model as written, centred)
and `(1, 0)` is the standardized innovation (non-centred) of a chain whose state
`t` given the previous one has mean `m_t` and standard deviation
`exp(log_scale_t)`. The log-Jacobian is `sum((1 - c_t) * log_scale_t)`.

At every restarting warm-up window, `phi_t` is refitted by least-squares
regression of `w_t` on `m_t` over the window's evidence, then `c_t` is chosen on
the grid `range(0, 1, 11)` by the default criterion of
[`candidate_scoring_losses`](@ref), the smallest position–gradient correlation,
with the gradients taken in the frame of the new `phi`. A chain thus ends up
centred where observations pin its states and close to the innovation where they
do not, such as in an unobserved stretch.

# Arguments

`log_scale`, `coefficient`, `drift` and `initial` are each a `Real` constant or
a callable applied to the whole source position vector, as a
[`Reparametrization`](@ref)'s location and log-scale are. A callable may return
one number for every state, or a vector with one entry per state: a `log_scale`
whose first entry is the stationary scale of an AR(1) series, or a
`coefficient` and `log_scale` for an OU process on an irregular time grid. The
first entries of `coefficient` and `drift` are not read; `initial` is a number.

They may read only coordinates that no chain or pair transforms, such as a log
step size, an autoregression coefficient or a mean. They are evaluated once per
chain on every path, and they run under AD on every gradient evaluation, so keep
them cheap and type-generic, as for a `Reparametrization`.

`phi` and `c` are the starting controls, one number for every state or a vector
with one entry per state; the default is the identity. `nonlinear_adapt = false`
keeps them fixed. Adaptation writes the controls in place: after an adaptive run,
`chain.phi` and `chain.c` hold them, and checkpoints record them under
`reparam_sources`.

# Examples

A random walk with a sampled log step size at coordinate 601, and an AR(1)
series around a mean, with a stationary first state:

```julia
walk = MarkovChain(1:600; log_scale = x -> x[601])
rho(x) = tanh(x[702])
ar1 = MarkovChain(602:700; coefficient = rho, drift = x -> (1 - rho(x)) * x[703],
    initial = x -> x[703],
    log_scale = x -> [x[701] - log1p(-rho(x)^2) / 2; fill(x[701], 98)])
ir = IndexedReparametrization(pairs; chains = [walk])
```
"""
struct MarkovChain{L,R,D,I}
    states::Vector{Int}
    log_scale::L
    coefficient::R
    drift::D
    initial::I
    phi::Vector{Float64}
    c::Vector{Float64}
end

function MarkovChain(states; log_scale, coefficient=1.0, drift=0.0, initial=0.0,
                     phi=0.0, c=1.0)
    s = collect(Int, states)
    isempty(s) && throw(ArgumentError("a MarkovChain needs at least one state"))
    allunique(s) || throw(ArgumentError(
        "a MarkovChain's states must be distinct coordinates; got a repeated one in $s"))
    for (name, a) in ((:log_scale, log_scale), (:coefficient, coefficient),
                      (:drift, drift), (:initial, initial))
        _is_accessor(a) || throw(ArgumentError(
            "MarkovChain: `$name` is a $(typeof(a)), which is neither a Real constant " *
            "nor callable. Pass a Real or a callable applied to the whole source " *
            "position vector, such as `x -> x[9]`."))
    end
    MarkovChain(s, log_scale, coefficient, drift, initial,
                _chain_controls(phi, length(s), :phi), _chain_controls(c, length(s), :c))
end

function _chain_controls(value, n, name)
    v = value isa Real ? fill(Float64(value), n) : collect(Float64, value)
    length(v) == n || throw(DimensionMismatch(
        "MarkovChain: `$name` has $(length(v)) entries for $n states"))
    all(isfinite, v) || throw(ArgumentError("MarkovChain: `$name` must be finite"))
    v
end

Base.copy(chain::MarkovChain) = MarkovChain(chain.states, chain.log_scale,
    chain.coefficient, chain.drift, chain.initial, copy(chain.phi), copy(chain.c))

# --- Validation, as part of `IndexedReparametrization`'s constructor -------------

_check_chains(pairs, ::Tuple{}) = nothing
function _check_chains(pairs, chains)
    chains isa Union{Tuple,AbstractVector} && all(c -> c isa MarkovChain, chains) ||
        throw(ArgumentError(
            "IndexedReparametrization: `chains` must be a vector or tuple of " *
            "MarkovChain; got a $(typeof(chains))"))
    chains isa AbstractVector && !isempty(chains) && !isconcretetype(eltype(chains)) &&
        throw(ArgumentError(
            "IndexedReparametrization chains must have a concrete element type " *
            "(got $(eltype(chains))): give the chains of one vector the same accessor " *
            "types, or pass chains of different types as a tuple"))
    owner = Dict{Int,Int}()
    for (n, chain) in enumerate(chains), idx in chain.states
        haskey(owner, idx) && throw(ArgumentError(
            "IndexedReparametrization: coordinate $idx is a state of chains " *
            "$(owner[idx]) and $n; a coordinate belongs to one chain"))
        owner[idx] = n
    end
    for (idx, _) in pairs
        idx isa Integer && haskey(owner, idx) && throw(ArgumentError(
            "IndexedReparametrization: coordinate $idx is a state of chain " *
            "$(owner[idx]) and also has a pair; a chain's states take no pairs"))
    end
    nothing
end

# --- The map ------------------------------------------------------------------
#
# Every path evaluates a chain's accessors once (`_chain_frames`) and walks its
# states in time order with `_chain_step` (source to model) or `_chain_unstep`
# (model to source), so the map, its inverse, the Enzyme objectives and the
# selection share one formula.

@inline _chain_frames(chain::MarkovChain, x) = (
    maybecall(chain.log_scale, x), maybecall(chain.coefficient, x),
    maybecall(chain.drift, x), maybecall(chain.initial, x))
@inline _per_state(value::Real, t) = value
@inline _per_state(value::AbstractVector, t) = value[t]
# State `t`'s location, from the previous state's model value.
@inline _chain_location(frames, t, previous) = t == 1 ? frames[4] :
    _per_state(frames[2], t) * previous + _per_state(frames[3], t)
@inline _chain_log_step(c, frames, t) = (1 - c) * _per_state(frames[1], t)
# The model value of state `t` and its log-Jacobian term.
@inline function _chain_step(phi, c, frames, t, previous, xt)
    log_step = _chain_log_step(c, frames, t)
    log_step, phi * _chain_location(frames, t, previous) + xt * exp(log_step)
end
# The source value of state `t` and its inverse log-Jacobian term.
@inline function _chain_unstep(phi, c, frames, t, previous, yt)
    log_step = _chain_log_step(c, frames, t)
    -log_step, (yt - phi * _chain_location(frames, t, previous)) * exp(-log_step)
end

_is_identity(chain::MarkovChain) = all(iszero, chain.phi) && all(isone, chain.c)

# Chains of different types come as a tuple. Every path that walks them goes
# through a tuple element by element (recursion, or `map`, unrolled at compile
# time), so each chain's code is type-stable, as AD needs; a vector of chains has
# one concrete type and is looped over.
_each_chain(f, chains::Tuple) = (map(f, chains); nothing)
_each_chain(f, chains) = foreach(f, chains)
_fold_chains(f, value, ::Tuple{}) = value
_fold_chains(f, value, chains::Tuple) = _fold_chains(f, f(value, first(chains)), Base.tail(chains))
function _fold_chains(f, value, chains)
    for chain in chains
        value = f(value, chain)
    end
    value
end

# Write the chains' states of `y` from `x` (source to model) and add their
# log-Jacobian to `ljac`.
_chains_forward!(ljac, y, chains, x) =
    _fold_chains((ljac, chain) -> _chain_forward_or_copy!(ljac, y, chain, x), ljac, chains)
function _chain_forward_or_copy!(ljac, y, chain, x)
    if _is_identity(chain)
        for idx in chain.states
            y[idx] = x[idx]
        end
        ljac
    else
        ljac + _chain_forward!(y, chain, chain.phi, chain.c, x, _chain_frames(chain, x))
    end
end
function _chain_forward!(y, chain, phi, c, x, frames)
    ljac = zero(eltype(y))
    previous = zero(eltype(y))
    for (t, idx) in enumerate(chain.states)
        log_step, yt = _chain_step(phi[t], c[t], frames, t, previous, x[idx])
        y[idx] = yt
        ljac += log_step
        previous = yt
    end
    ljac
end

# Recover the chains' states of `source` from `target` (model to source). Their
# accessors read coordinates nothing transforms, the same in either frame; they
# are read off `target`.
_chains_inverse!(ljac, source, chains, target) =
    _fold_chains((ljac, chain) -> _chain_inverse_or_copy!(ljac, source, chain, target), ljac, chains)
function _chain_inverse_or_copy!(ljac, source, chain, target)
    if _is_identity(chain)
        for idx in chain.states
            source[idx] = target[idx]
        end
        ljac
    else
        ljac + _chain_inverse!(source, chain, chain.phi, chain.c, target,
                               _chain_frames(chain, target))
    end
end
function _chain_inverse!(source, chain, phi, c, target, frames)
    ljac = zero(eltype(source))
    states = chain.states
    for (t, idx) in enumerate(states)
        previous = t == 1 ? zero(eltype(target)) : target[states[t - 1]]
        log_step, st = _chain_unstep(phi[t], c[t], frames, t, previous, target[idx])
        source[idx] = st
        ljac += log_step
    end
    ljac
end

# --- Controls --------------------------------------------------------------------

_chain_controls(chain::MarkovChain) = (; phi=copy(chain.phi), c=copy(chain.c))
function _restore_chain_controls!(chain::MarkovChain, controls, n)
    length(controls.phi) == length(controls.c) == length(chain.states) ||
        throw(ArgumentError(
            "chain $n has $(length(chain.states)) states but the controls recorded " *
            "$(length(controls.phi)); build the chains exactly as the run was given them"))
    chain.phi .= controls.phi
    chain.c .= controls.c
    chain
end
_snapshot_chains(chains::Tuple) = map(copy, chains)
_snapshot_chains(chains::AbstractVector) = map(copy, chains)
_restore_chains!(chains, old) =
    foreach((chain, o) -> (chain.phi .= o.phi; chain.c .= o.c), chains, old)

# --- Selection ---------------------------------------------------------------------
#
# The evidence is a set of weighted source-frame columns with their gradients. For
# a column, the chain's model values `w_t` and locations `m_t` follow from the
# current controls `(phi0, c0)`. The gradient of state `t`'s source coordinate is
# `a_t * exp((1 - c0_t) * log_scale_t)`, where `a_t` is the derivative in `w_t`
# through every later state, `a_t = v_t + phi0_{t+1} * coefficient_{t+1} * a_{t+1}`,
# and `v_t` is the model's own derivative in `w_t`. Undoing that recursion gives
# `v`; redoing it with the new `phi` gives the gradient in the new frame, which is
# what the candidates for `c` are scored on. Each column costs one pass over the
# chain per direction, so a selection is linear in the chain's length.

const CHAIN_CENTERINGS = range(0, 1, 11)

_column_weight(::Nothing, k) = 1.0
_column_weight(weights, k) = Float64(weights[k])

# Model values `w` and locations `m` of one column under `(phi0, c0)`, and the
# derivative `a` in `w` through every later state under `phi`.
function _chain_column!(w, m, a, chain, phi0, c0, phi, x, g, frames)
    states = chain.states
    previous = 0.0
    for (t, idx) in enumerate(states)
        m[t] = _chain_location(frames, t, previous)
        _, w[t] = _chain_step(phi0[t], c0[t], frames, t, previous, x[idx])
        previous = w[t]
    end
    T = length(states)
    old_next = new_next = 0.0
    for t in T:-1:1
        a_old = g[states[t]] * exp(-_chain_log_step(c0[t], frames, t))
        a_new = if t == T
            a_old
        else
            step_coefficient = _per_state(frames[2], t + 1)
            a_old - (phi0[t + 1] * old_next - phi[t + 1] * new_next) * step_coefficient
        end
        a[t] = a_new
        old_next, new_next = a_old, a_new
    end
    nothing
end

# The regression slope of `w_t` on `m_t` for every state, and the candidates'
# loss accumulators for `c_t` in the frame of that slope. `phi_t` keeps its value
# where `m_t` does not vary. Returns `(phi, losses, groups)`: `losses[j, t]` is
# candidate `CHAIN_CENTERINGS[j]` of state `t`, and `groups` the number of
# positive-weight columns.
function _chain_scores(chain, positions, gradients, weights)
    T = length(chain.states)
    phi0, c0 = chain.phi, chain.c
    w, m, a = zeros(T), zeros(T), zeros(T)
    columns = [k for k in axes(positions, 2) if _column_weight(weights, k) > 0]
    regression = [WeightedReparametrizationLoss() for _ in 1:T]
    for k in columns
        x = view(positions, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, a, chain, phi0, c0, phi0, x, view(gradients, :, k), frames)
        for t in 1:T
            OnlineStatsBase.fit!(regression[t], (0.0, w[t], m[t]);
                                 weight=_column_weight(weights, k))
        end
    end
    phi = copy(phi0)
    for (t, r) in enumerate(regression)
        slope = r.co_position_gradient / r.m2_gradient
        r.m2_gradient > 0 && isfinite(slope) && (phi[t] = slope)
    end
    losses = [WeightedReparametrizationLoss() for _ in CHAIN_CENTERINGS, _ in 1:T]
    for k in columns
        x = view(positions, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, a, chain, phi0, c0, phi, x, view(gradients, :, k), frames)
        weight = _column_weight(weights, k)
        for t in 1:T
            log_scale = _per_state(frames[1], t)
            residual = w[t] - phi[t] * m[t]
            # The candidates' scales exp((1 - c) * log_scale) differ by one factor
            # per grid step: two `exp`s per state rather than two per candidate.
            factor = exp(step(CHAIN_CENTERINGS) * log_scale)
            scale = exp((1 - first(CHAIN_CENTERINGS)) * log_scale)
            for (j, c) in enumerate(CHAIN_CENTERINGS)
                OnlineStatsBase.fit!(losses[j, t],
                    ((1 - c) * log_scale, residual / scale, a[t] * scale); weight)
                scale /= factor
            end
        end
    end
    phi, losses, length(columns)
end

# Choose the chain's new controls from the evidence, as `_minimizing_source!` does
# for a pair: only finite losses compete, a state with none keeps its centering,
# and `excluded` collects `index => kept` for `_warn_excluded_candidates`. Returns
# whether the controls changed.
function _select_chain!(excluded, chain, positions, gradients, weights=nothing;
                        loss_kwargs...)
    phi, losses, groups = _chain_scores(chain, positions, gradients, weights)
    groups > 2 || return false
    c = copy(chain.c)
    for t in eachindex(c)
        values = [reparametrization_loss(losses[j, t]; loss_kwargs...)
                  for j in eachindex(CHAIN_CENTERINGS)]
        finite = any(isfinite, values)
        any(!isfinite, values) && push!(excluded, chain.states[t] => !finite)
        finite || continue
        c[t] = CHAIN_CENTERINGS[argmin(j -> isfinite(values[j]) ? values[j] : Inf,
                                       eachindex(values))]
    end
    changed = phi != chain.phi || c != chain.c
    chain.phi .= phi
    chain.c .= c
    changed
end

# Move the chain's states of every pool column from `(phi0, c0)` into the chain's
# current controls, positions and gradients, so that what later pairs score reads
# them in the new frame, as `_select_on_pool!` moves its pairs' coordinates.
function _move_chain_columns!(chain, phi0, c0, positions, gradients)
    T = length(chain.states)
    w, m, a = zeros(T), zeros(T), zeros(T)
    for k in axes(positions, 2)
        x, g = view(positions, :, k), view(gradients, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, a, chain, phi0, c0, chain.phi, x, g, frames)
        for (t, idx) in enumerate(chain.states)
            log_step = _chain_log_step(chain.c[t], frames, t)
            x[idx] = (w[t] - chain.phi[t] * m[t]) * exp(-log_step)
            g[idx] = a[t] * exp(log_step)
        end
    end
    nothing
end

# The pool search's chain step, before the pairs: each chain is selected on the
# pool and its states are moved into the new frame.
_select_chains_on_pool!(::Tuple{}, xg...; loss_kwargs...) = nothing
function _select_chains_on_pool!(chains, positions, gradients; loss_kwargs...)
    excluded = Pair{Any,Bool}[]
    for chain in chains
        phi0, c0 = copy(chain.phi), copy(chain.c)
        _select_chain!(excluded, chain, positions, gradients; loss_kwargs...) &&
            _move_chain_columns!(chain, phi0, c0, positions, gradients)
    end
    _warn_excluded_candidates(excluded)
    nothing
end

# --- Streaming evidence -------------------------------------------------------------
#
# A chain's candidates are scored in the frame of the `phi` the window ends with,
# which the leaves cannot know as they stream in, so a chain keeps a weighted
# sample of the window's leaves (the evidence a `WindowSelectionPlan` gets) and is
# selected on it at the window's end. The pairs keep their running accumulators.

struct ChainEvidence{O}
    pairs::O
    sample::WeightedLeafSample
end

_observe_nonlinear!(plan::Union{DirectCandidateScoring,CandidateScoringPlan}, ir,
                    online::ChainEvidence, position, gradient, weight) = begin
    _observe_nonlinear!(plan, ir, online.pairs, position, gradient, weight)
    _observe_sample!(online.sample, position, gradient, weight)
    nothing
end
_mark_group!(online::ChainEvidence) = _mark_group!(online.pairs)
reset!(online::ChainEvidence) = (reset!(online.pairs); reset!(online.sample); nothing)
_candidate_scoring_losses(online::ChainEvidence) = _candidate_scoring_losses(online.pairs)
_leaf_sample(online::ChainEvidence) = online.sample
_leaf_sample(online::WeightedLeafSample) = online
_leaf_sample(online) = nothing

function optimize!(ir::IndexedReparametrization, online::ChainEvidence; loss_kwargs=(;))
    excluded = Pair{Any,Bool}[]
    positions, gradients = _sample_evidence(online.sample)
    for chain in ir.chains
        _select_chain!(excluded, chain, positions, gradients; loss_kwargs...)
    end
    _warn_excluded_candidates(excluded)
    optimize!(ir, online.pairs; loss_kwargs)
end

# --- Retrospective loss rows ----------------------------------------------------------

function _chain_candidate_scoring_losses(chains, positions, gradients, weights)
    isnothing(weights) || for weight in weights
        weight >= 0 || throw(ArgumentError(
            "nonlinear observation weight must be nonnegative, got $weight"))
    end
    rows = NamedTuple[]
    for (chain_number, chain) in enumerate(chains)
        phi, losses, groups = _chain_scores(chain, positions, gradients, weights)
        for (t, index) in enumerate(chain.states), (j, c) in enumerate(CHAIN_CENTERINGS)
            accumulator = losses[j, t]
            push!(rows, (; chain_number, state_number=t, index, phi=phi[t], candidate=c,
                loss=groups > 2 ? reparametrization_loss(accumulator) : missing,
                groups,
                effective_n=iszero(accumulator.weight2) ? missing : effective_n(accumulator)))
        end
    end
    rows
end
