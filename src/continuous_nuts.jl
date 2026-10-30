# ============================================================================
# Continuously adapting NUTS for a target that changes between transitions.
#
# `FixedKernelNUTSState` / `fixed_kernel_nuts_step` (interruptible_mcmc.jl)
# advance a changing target under a kernel learned ONCE: when the target moves
# on, as SAEM's conditional posteriors p(η_i | y_i, θ, σ, Ω) do at every
# iteration, that kernel mixes worse and worse, and the only remedy is a
# deliberate re-warm (snag `continuous-nuts-15212ad7`, reporter NLMEEstimation).
# This sampler keeps adapting instead. There is no warm-up phase, no window,
# no restart and no freeze: every transition updates the diagonal metric and the
# step size from what that transition saw, with exponential forgetting, for as
# long as the caller keeps stepping.
#
# What it borrows from the other samplers (user direction, 2026-10-30: "take
# from whmc whatever is helpful"):
#   * the start: `initialize_mcmc` — Pathfinder from a vector or random start,
#     or a caller-supplied `(; position, squared_scale)` — and DynamicHMC's
#     initial step-size search, as `adaptive_warmup_mcmc` and the cooperative
#     chains do;
#   * the metric: the Nutpie diagonal scale `(var(position) / var(gradient))^(1/4)`
#     of the good leapfrog leaves (`_fit_leaves!`'s filter), the estimate the
#     cooperative sampler fits from its pooled moments, here over an
#     exponentially forgotten window of transitions (`WeightedScaleAdaptation`
#     plus `_forget!`), regularized toward the start scale by a pseudo-count
#     that forgets at the same rate;
#   * the step size: the cooperative sampler's κ fit, acceptance ≈ (1 − π)·2Φ(−κε²)
#     over the recent (ε, acceptance) pairs with decaying weights and a backoff
#     on failed transitions.
# What it does differently: the leaves are not stored. `LeafMomentsPosterior`
# feeds each good leaf into the moments as DynamicHMC visits it, so a transition
# costs O(d) memory beyond the trajectory itself, at any tree depth.
#
# Everything here is linear in the dimension (user rule, decision `1o12joq`):
# the state holds seven length-d vectors and no d × d object; the metric is a
# `Diagonal`. The successive-reflections metric is a later increment.
# ============================================================================

"Forgetting half-life of the metric evidence, in transitions."
const _DEFAULT_CONTINUOUS_HALF_LIFE = 20.0
"Pseudo-count of leapfrog states the start scale is worth; it forgets with the evidence."
const _DEFAULT_CONTINUOUS_PRIOR_WEIGHT = 10.0
# The κ fit: (ε, acceptance) pairs kept, their geometric weighting, the prior
# weight toward the previous κ, and the grid sizes (`_fit_log_kappa`). The fit
# runs on every transition, so the grids are coarser than the cooperative
# sampler's batch-end ones; 25 coarse points over ±3 in log κ resolve 0.25,
# and the 11 fine points then 0.05, i.e. 1.3 % in ε. The prior weight is five
# times the cooperative sampler's: there the prior is a pooled κ and the fit
# runs once per batch, here it is the chain's own last κ and the fit runs on
# every transition, so it is what damps the step size's noise. Measured on
# stationary nested Gaussian and Poisson conditionals (d = 250 to 4,000), a
# weight of 10 against 2 cut the step size's log-sd from 0.10 to 0.03 and
# raised the median ESS per gradient evaluation by up to 2.3× (snag
# `continuous-nuts-15212ad7`); a longer history (32, 64) did not help.
const _CONTINUOUS_KAPPA_HISTORY = 16
const _CONTINUOUS_KAPPA_DECAY = 0.9
const _CONTINUOUS_KAPPA_PRIOR_WEIGHT = 10.0
const _CONTINUOUS_KAPPA_COARSE = 25
const _CONTINUOUS_KAPPA_FINE = 11

"""
    LeafMomentsPosterior(posterior, moments)

The target a continuous NUTS transition runs against: `posterior`, whose every
good leapfrog leaf (not the initial point, energy error above `_GOOD_LEAF_MIN_DH`)
is fed into `moments` as DynamicHMC visits it. Nothing is stored per leaf.
"""
struct LeafMomentsPosterior{P,M} <: LeafObservingPosterior{P}
    posterior::P
    moments::M
end
Base.parent(p::LeafMomentsPosterior) = p.posterior

observe_leaf!(p::LeafMomentsPosterior, z, Δ, is_initial) = begin
    (!is_initial && isfinite(Δ) && Δ > _GOOD_LEAF_MIN_DH) || return nothing
    OnlineStatsBase.fit!(p.moments, z.Q.q, z.Q.∇ℓq)
    nothing
end

"""
    ContinuousNUTSState(rng, lpdf; init=missing, target_acceptance_rate=0.8,
                        max_tree_depth=10, half_life=20, prior_weight=10,
                        pathfinder_kw=(;), blas_threads=1)

In-memory state for [`continuous_nuts_step`](@ref): a NUTS chain whose
diagonal metric and step size keep adapting on every transition, for a target
that changes between transitions. Initialization is
[`adaptive_warmup_mcmc`](@ref)'s: `init` takes every form documented there, so
by default Pathfinder runs from a random start on `lpdf`, a vector `init` runs
Pathfinder from it, and `init = (; position, squared_scale)` starts exactly
there with that diagonal covariance. The start position and the diagonal of
the initializer's covariance give the first metric; DynamicHMC's initial
step-size search gives the first step size. Construction evaluates `lpdf` for
that search and for Pathfinder, nothing else.

`half_life` is the memory of the metric evidence, in transitions: the leapfrog
states of a transition `half_life` transitions ago weigh half as much as the
current one's (`Inf` never forgets). `prior_weight` is how many leapfrog states
the start scale is worth; it forgets at the same rate. `target_acceptance_rate`
is the step size's target. `pathfinder_kw` is forwarded to the Pathfinder
initializer. `blas_threads` is held for the construction as for every sampler.

Fields a caller reads: `position` (the current position, in the target's
coordinates), `scale` (the metric, a `Diagonal`; `scale * scale'` is the
estimated posterior covariance), `stepsize`, `rng`, `n_transitions`,
`total_evaluation_counter` and `n_divergent`. The state is advanced in place
by [`continuous_nuts_step`](@ref) and moved by [`reposition!`](@ref).
"""
mutable struct ContinuousNUTSState{R,K,A}
    const rng::R
    const dimension::Int
    const algorithm::A
    const target_acceptance_rate::Float64
    const decay::Float64                              # per-transition forgetting factor
    const moments::WeightedScaleAdaptation{Float64}   # good leapfrog leaves, forgotten
    const prior_scale::Vector{Float64}                # the start scale
    position::Vector{Float64}
    scale::Diagonal{Float64,Vector{Float64}}
    scale_buffer::Vector{Float64}                     # the next scale is written here
    kinetic_energy::K
    stepsize::Float64
    prior_weight::Float64
    log_kappa::Float64
    backoff::Float64
    const stepsizes::Vector{Float64}                  # the last `_CONTINUOUS_KAPPA_HISTORY` pairs
    const acceptances::Vector{Float64}
    n_transitions::Int
    total_evaluation_counter::Int
    n_divergent::Int
end

function ContinuousNUTSState(rng, lpdf; init=missing, target_acceptance_rate::Real=0.8,
        max_tree_depth::Integer=10, half_life::Real=_DEFAULT_CONTINUOUS_HALF_LIFE,
        prior_weight::Real=_DEFAULT_CONTINUOUS_PRIOR_WEIGHT, pathfinder_kw=(;),
        blas_threads=_DEFAULT_BLAS_THREADS)
    0 < target_acceptance_rate < 1 || throw(ArgumentError(
        "`target_acceptance_rate` must lie strictly between 0 and 1; got $target_acceptance_rate."))
    max_tree_depth >= 1 || throw(ArgumentError(
        "`max_tree_depth` must be positive; got $max_tree_depth."))
    half_life > 0 || throw(ArgumentError(
        "`half_life` must be positive (in transitions; `Inf` never forgets); got $half_life."))
    prior_weight >= 0 && isfinite(prior_weight) || throw(ArgumentError(
        "`prior_weight` must be a finite nonnegative pseudo-count; got $prior_weight."))
    _with_blas_threads(blas_threads) do
        rng = copy(rng)
        dimension = LogDensityProblems.dimension(lpdf)
        (; position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress=nothing, pathfinder_kw...)
        scale = _initial_diagonal_scale(squared_scale)
        q = collect(Float64, position)
        kinetic_energy = _stream_kinetic_energy(scale)
        # A start outside the support cannot be stepped from, and says so.
        pg = _evaluate_transition_start(lpdf, q, "the initial position")
        stepsize = DynamicHMC.find_initial_stepsize(
            DynamicHMC.InitialStepsizeSearch(),
            DynamicHMC.local_log_acceptance_ratio(
                DynamicHMC.Hamiltonian(kinetic_energy, lpdf),
                DynamicHMC.PhasePoint(pg, DynamicHMC.rand_p(rng, kinetic_energy)),
            ),
        )
        target = Float64(target_acceptance_rate)
        ContinuousNUTSState(
            rng, dimension, DynamicHMC.NUTS(; max_depth=max_tree_depth), target,
            isfinite(half_life) ? 2.0^(-1 / half_life) : 1.0,
            WeightedScaleAdaptation(dimension), copy(scale.diag),
            q, scale, similar(scale.diag), kinetic_energy, stepsize, Float64(prior_weight),
            _kappa_log(stepsize, target), 1.0, Float64[], Float64[], 0, 0, 0,
        )
    end
end

"""
    continuous_nuts_step(state::ContinuousNUTSState, lpdf; adapt=true, blas_threads=1)

Take one NUTS transition under the supplied `LogDensityProblems` target and,
with `adapt=true`, adapt the state's diagonal metric and step size to what the
transition saw. Return `(; state, divergent, stats)`: `state` is the same
object, advanced in place; `divergent` is this transition's divergence flag;
`stats` is DynamicHMC's tree statistics (`steps` counts the gradient
evaluations). Read the new position from `state.position`.

The target may change on every call, but must keep the same dimension and
coordinate frame. Its log density and gradient are freshly evaluated at the
saved position (the last transition's, or the one [`reposition!`](@ref)
stored) before sampling; a position the new target puts outside its
support (log density `-Inf`) is an `ArgumentError` saying so, and a `NaN` or
`+Inf` log density or a non-finite gradient there fails DynamicHMC's
evaluation. At a leapfrog position, leaving the support is an ordinary
divergence, and the gradient at such a point is never read.

Adaptation, per transition: the evidence forgets by the state's half-life;
every good leapfrog leaf of the transition feeds position and gradient moments;
the metric becomes the diagonal scale `(var(position) / var(gradient))^(1/4)`
of those moments, shrunk toward the start scale by the remaining prior weight;
the step size and its (ε, acceptance) history are carried through the metric
change by the geometric mean of the scale change; then the step size moves
toward `target_acceptance_rate` through the κ fit of the recent pairs, halved
after a transition that accepted less than 10 %. `adapt=false` takes the
transition under the current kernel and changes nothing but the position: the
same draw a [`FixedKernelNUTSState`](@ref) with that kernel takes.

Positions from different targets are not a stationary posterior sample. The
transition runs with `blas_threads` BLAS threads and restores the caller's
count afterwards; `nothing` leaves BLAS untouched.
"""
continuous_nuts_step(state::ContinuousNUTSState, lpdf; adapt::Bool=true,
        blas_threads=_DEFAULT_BLAS_THREADS) = _with_blas_threads(blas_threads) do
    dimension = LogDensityProblems.dimension(lpdf)
    state.dimension == dimension || throw(DimensionMismatch(
        "state holds a $(state.dimension)-dimensional position but the " *
        "supplied lpdf has dimension $dimension."))
    pg = _evaluate_transition_start(lpdf, state.position, "the state's position")
    if adapt
        _forget!(state.moments, state.decay)
        state.prior_weight *= state.decay
    end
    target = adapt ? LeafMomentsPosterior(lpdf, state.moments) : lpdf
    ε = state.stepsize
    hamiltonian = DynamicHMC.Hamiltonian(state.kinetic_energy, target)
    next_pg, stats = DynamicHMC.sample_tree(state.rng, state.algorithm, hamiltonian, pg, ε)
    state.position = copy(next_pg.q)
    state.n_transitions += 1
    state.total_evaluation_counter += stats.steps
    divergent = DynamicHMC.is_divergent(stats.termination)
    divergent && (state.n_divergent += 1)
    adapt && _adapt_continuous!(state, ε, stats.acceptance_rate)
    (; state, divergent, stats)
end

"""
    reposition!(state::ContinuousNUTSState, position::AbstractVector)

Translate the chain to `position` and leave its kernel alone. A caller that
moves the target and the chain together calls it instead of building a new
state: SAEM's mu-referenced re-centring shifts each subject's η after an M-step
that moved θ, so that φ = Dθ + η stays where it was, and a translation of the
target leaves the metric and step size valid. The saved position and the mean
of the position evidence move by the same vector, so the leapfrog states the
metric is estimated from are translated with the chain (their spread, the
gradient evidence, and so the metric, are what they were); `scale`, `stepsize`,
the forgetting, the κ fit, `rng`, `n_transitions` and the counters stay. A
translated target sampled by the translated chain is therefore the same chain,
draw for draw up to rounding. Rebuilding the state through
`init = (; position, squared_scale)` would instead restart the metric evidence
from `prior_weight` pseudo-states and run DynamicHMC's step-size search again,
and leaving the position evidence where it was would inflate the metric by the
move until it is forgotten.

`position` is copied (any real `AbstractVector` of length `state.dimension`; a
wrong length is a `DimensionMismatch`, a non-finite entry an `ArgumentError`,
and a refused move changes nothing) and not evaluated: the next
[`continuous_nuts_step`](@ref) evaluates it under the target it is given, and a
position that target excludes is that call's `ArgumentError`. Returns `state`.
"""
function reposition!(state::ContinuousNUTSState, position::AbstractVector)
    length(position) == state.dimension || throw(DimensionMismatch(
        "`position` has length $(length(position)) but the state holds a " *
        "$(state.dimension)-dimensional position."))
    q = collect(Float64, position)
    all(isfinite, q) || throw(ArgumentError(
        "`position` must contain only finite values; a NUTS transition cannot start from it."))
    # The evidence moves with the chain: every leapfrog state it holds is
    # translated by the same vector, which shifts their mean and nothing else
    # (`fit!` keeps the squared deviations about the mean).
    state.moments.position_mean .+= q .- state.position
    state.position = q
    state
end

# The next diagonal scale into `out`: the Nutpie estimate of the forgotten
# moments, with the start scale `s₀` entering as `prior_weight` pseudo-states of
# variance `s₀²` and gradient variance `1 / s₀²` (so with no evidence the scale is
# `s₀`, and the ratio needs no denominator). A coordinate the evidence cannot
# judge (a non-finite or non-positive ratio) keeps its current scale. Returns
# the mean log change, `mean(log(out) - log(current))`.
function _continuous_scales!(out, a::WeightedScaleAdaptation, prior_scale, prior_weight, current)
    log_change = 0.0
    @inbounds for i in eachindex(out, current, prior_scale)
        s0 = prior_scale[i]
        num = a.position_m2[i] + prior_weight * abs2(s0)
        den = a.gradient_m2[i] + prior_weight / abs2(s0)
        s = (num / den)^(1 / 4)
        if isfinite(s) && s > 0
            out[i] = s
            log_change += log(s) - log(current[i])
        else
            out[i] = current[i]
        end
    end
    log_change / length(out)
end

_adapt_continuous!(state::ContinuousNUTSState, ε, acceptance) = begin
    # 1. The metric, from the moments the transition just fed.
    old = state.scale.diag
    new = state.scale_buffer
    log_change = _continuous_scales!(new, state.moments, state.prior_scale, state.prior_weight, old)
    state.scale = Diagonal(new)
    state.scale_buffer = old
    state.kinetic_energy = _stream_kinetic_energy(state.scale)
    # 2. Carry the step size through the metric change. A uniform growth of the
    #    scale by `c` lengthens every leapfrog step by `c` at a fixed ε, so ε
    #    and the pairs the κ fit reads (all measured under earlier metrics) are
    #    divided by the geometric mean growth, and κ ∝ 1/ε² follows.
    if log_change != 0
        carry = exp(-log_change)
        state.stepsizes .*= carry
        ε *= carry
        state.log_kappa += 2 * log_change
    end
    # 3. The step size, from the recent (ε, acceptance) pairs.
    push!(state.stepsizes, ε)
    push!(state.acceptances, acceptance)
    if length(state.acceptances) > _CONTINUOUS_KAPPA_HISTORY
        popfirst!(state.stepsizes)
        popfirst!(state.acceptances)
    end
    # A divergence alone is not a step-size failure: leaving the support
    # diverges at any step size (the cooperative sampler's rule).
    failed = acceptance < 0.1
    state.backoff = failed ? state.backoff / 2 : min(1.0, 1.5 * state.backoff)
    n = length(state.acceptances)
    weights = _CONTINUOUS_KAPPA_DECAY .^ (n-1:-1:0)
    log_kappa = _fit_log_kappa(state.stepsizes, state.acceptances, weights,
        state.log_kappa, state.log_kappa, _CONTINUOUS_KAPPA_PRIOR_WEIGHT;
        coarse_points=_CONTINUOUS_KAPPA_COARSE, fine_points=_CONTINUOUS_KAPPA_FINE)
    stepsize = _kappa_stepsize(log_kappa, state.target_acceptance_rate) * state.backoff
    if isfinite(stepsize) && stepsize > 0
        state.log_kappa = log_kappa
        state.stepsize = stepsize
    else
        state.stepsize = ε
    end
    state
end
