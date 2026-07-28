# Generic adaptive nonlinear reparametrization machinery
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

maybecall(f::Function, args...; kwargs...) = f(args...; kwargs...)
maybecall(x, args...; kwargs...) = x

"""
    ReparametrizedProblem(reparametrizer, problem, ad_backend=nothing)

Wrap a `LogDensityProblems`-compatible `problem` in a nonlinear reparametrization
that warm-up is allowed to ADAPT.

`reparametrizer` is an [`IndexedReparametrization`](@ref). The sampler works in
its *source* coordinates; `logdensity` maps a source-coordinate position `x` into
the coordinates `problem` is written in and adds the log-Jacobian:

```julia
ljac, y = reparametrizer(x)
logdensity(rp, x) == ljac + logdensity(problem, y)
```

`dimension` and `capabilities` are forwarded to `problem` unchanged.

!!! warning "Nothing happens unless you build the reparametrization yourself"
    An empty [`IndexedReparametrization`](@ref) is a no-op — and an empty one is
    exactly what every other problem reports, since
    `WarmupHMC.reparametrizer(::Any) = IndexedReparametrization([])`. So
    `nonlinear_adapt=true` (the default on every sampler) changes nothing at all
    on a plain log-density problem. There is no automatic detection of
    hierarchical structure: you supply the coordinate indices and the
    location/log-scale accessors, or nothing is reparametrized.

# Gradients

`logdensity_and_gradient` differentiates only through the reparametrization and
reuses `problem`'s own gradient, so an inner problem with a native gradient (a
BridgeStan model, say) keeps providing the expensive part. Writing
`L(x) = ljac(x) + ld(y(x))`,

```
∂L/∂x = ∂ljac/∂x + (∂y/∂x)' * ∂ld/∂y
```

is obtained by AD-ing the scalar `x -> ljac(x) + dot(g_y, y(x))` with `g_y` held
fixed at the inner gradient — one call to the inner problem plus one AD pass over
the transform, per gradient evaluation. The transform is therefore on the
gradient hot path, and the accessor closures in each
[`Reparametrization`](@ref) run under AD.

`ad_backend` is a DifferentiationInterface.jl backend. DifferentiationInterface
is a hard dependency of WarmupHMC, so the *interface* is always there, but a
backend object only works once you load the AD package behind it — `AutoEnzyme`
needs `using Enzyme`. Constructing the backend object alone is not enough.

A reverse-mode backend is the reasonable default here — but that follows from an
operation count, and the measured wall-clock does not follow the operation count.
The objective differentiated here is scalar in the *full* parameter vector, so
forward mode costs `ceil(n / chunksize)` sweeps of the transform per gradient
while reverse mode costs one.

!!! warning "That argument does not predict wall-clock, and gets the direction wrong"
    It is tempting to conclude that the gap widens with dimension — exactly
    where reparametrization is worth doing. **Measured, it narrows and then
    reverses.** Enzyme with `Const` divided by ForwardDiff, per wrapped
    gradient (below 1.0 = Enzyme faster), across five processes with the
    backend order rotated per round:

    | target | `d` | Enzyme ÷ ForwardDiff |
    |---|---|---|
    | `funnel`                   | 10 | 0.35–0.42× |
    | `eight_schools`            | 10 | 0.44–0.92× |
    | `seeds`                    | 26 | 2.31–5.42× |
    | `radon_variable_intercept` | 89 | 1.21–1.39× |
    | `radon_partially_pooled`   | 88 | 1.25–1.52× |

    Reverse mode wins on the two *smallest* targets and loses on the three
    larger ones. Sampling is unaffected either way — ESS per 1000 gradients,
    gradient counts, the fitted `c` and stuck-adaptation counts are identical
    across backends; the backend sets the cost of a gradient, not how many are
    needed.

    So: pick the backend by measuring your own target, not by dimension. Neither
    mode is the universally correct default, and this docstring previously
    claimed one was.

    Measured at `b5c7dee`, 184 runs per backend, results checked in at
    `068cdeb`. Two known artifacts: DifferentiationInterface re-prepares on
    every call (10–16% of the call at `d ≈ 88`, and equal under both backends —
    reusing a prep object measured neutral-to-worse), and running a sampler
    before timing warms the ForwardDiff path enough to make a naive
    microbenchmark ~2× kinder to it.

!!! warning "Enzyme needs `function_annotation = Enzyme.Const`, and its own error message points the wrong way"
    A bare `AutoEnzyme()` **does not work here**. The differentiated objective is
    a closure over this `ReparametrizedProblem` — it captures the reparametrizer
    and the frozen inner gradient `g_y` — and Enzyme cannot prove that captured
    state is read-only:

        EnzymeMutabilityException: Function argument passed to autodiff cannot be proven readonly

    Pass `function_annotation = Enzyme.Const`. That is semantically exact rather
    than a workaround: `g_y` is frozen by construction, and the derivative taken
    is with respect to the *parameter vector*, never with respect to the problem.

    Enzyme's own error text suggests `function_annotation = Enzyme.Duplicated`.
    **Do not take that hint.** It is correct — it agrees with `Const` to ≤9.1e-13
    — but it allocates and propagates a shadow copy of the closure on every call.
    The hint diagnoses the problem; it is not the fix.

    How much that costs is **target-dependent**, and earlier revisions of this
    docstring published a single ratio that does not generalize. `Duplicated`
    divided by `Const`, per wrapped gradient:

    | target | `d` | `Duplicated` ÷ `Const` |
    |---|---|---|
    | `funnel`                   | 10 | 11.3–13.6× |
    | `eight_schools`            | 10 | 3.8–4.9×   |
    | `seeds`                    | 26 | 0.96–1.16× |
    | `radon_variable_intercept` | 89 | 1.01–1.05× |
    | `radon_partially_pooled`   | 88 | 1.02–1.06× |

    The shadow copy is a roughly **fixed per-call cost** — about 5 µs at `d = 10`,
    about 33 µs at `d ≈ 88` — so it dominates when the gradient is otherwise
    cheap and disappears when it is not. The funnel is the extreme case, not a
    representative one; funnel measurements at different `c` have landed anywhere
    from ~10× to ~22×, which is why no single number belongs here.

    None of that changes the recommendation. `Const` is the right annotation on
    **correctness** grounds everywhere — `g_y` is frozen by construction — and it
    is never slower. It is merely not always dramatically faster.

    Measured at `b5c7dee`, 184 runs per backend, results checked in at `068cdeb`.

A backend is required in practice, and omitting it fails *late*: the
two-argument constructor `ReparametrizedProblem(r, p)` stores `nothing`, which
is not a backend, so `logdensity` keeps working on that object and the first
`logdensity_and_gradient` call `MethodError`s inside `value_and_gradient`.
Construction itself never complains.

# Example

```julia
using WarmupHMC, DifferentiationInterface, Enzyme

# Coordinates 2:11 are the group effects; their location is fixed at 0 and their
# log-scale is half of coordinate 1 (Neal's funnel, `xᵢ ~ Normal(0, exp(v/2))`).
ir = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                           0., x -> x[1] / 2)
    for i in 2:11
])
rp = ReparametrizedProblem(ir, my_problem,
                           AutoEnzyme(; function_annotation = Enzyme.Const))
result = adaptive_warmup_mcmc(rng, rp)
```

Returned draws are in the wrapped problem's own parametrization: warm-up applies
the transform to `posterior_position` before returning. This holds under
`nonlinear_adapt=false` too — that flag gates whether the centering is *fitted*,
never which frame the result is reported in, since the sampler works in the
reparametrizer's source frame either way.

See [Nonlinear reparametrization](@ref) for a runnable end-to-end version.
"""
struct ReparametrizedProblem{R,P,B}
    reparametrizer::R
    problem::P
    ad_backend::B
end
ReparametrizedProblem(r, p) = ReparametrizedProblem(r, p, nothing)
reparametrizer(p::ReparametrizedProblem) = p.reparametrizer
reparametrizer(p::WrappedLogDensityProblem) = reparametrizer(parent(p))
reparametrizer(::Any) = IndexedReparametrization([])
LogDensityProblems.capabilities(::Type{<:ReparametrizedProblem{R,P}}) where {R,P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::ReparametrizedProblem) = LogDensityProblems.dimension(p.problem)
LogDensityProblems.logdensity(p::ReparametrizedProblem, x::AbstractVector) = begin
    ljac, y = p.reparametrizer(x)
    ljac + LogDensityProblems.logdensity(p.problem, y)
end
LogDensityProblems.logdensity_and_gradient(p::ReparametrizedProblem, x::AbstractVector) = begin
    # Differentiate only through the reparametrization, not through the inner problem.
    # The inner problem (e.g. BridgeStan) provides its own gradients via FFI.
    _logdensity_and_gradient_reparam(p, x)
end
"""
    WarmupHMC._logdensity_and_gradient_reparam(p::ReparametrizedProblem, x)

The gradient of a [`ReparametrizedProblem`](@ref); see that docstring for the
user-facing contract.

Differentiate ONLY through the reparametrization transform (pure Julia) and reuse
the inner problem's own `logdensity_and_gradient` — which may be an FFI call
(BridgeStan) that no Julia AD backend could differentiate through anyway.

    L(x)   = ljac(x) + ld(y(x))
    ∂L/∂x  = ∂ljac/∂x + (∂y/∂x)' ∂ld/∂y

`∂ld/∂y` comes from the inner problem; the rest comes from AD over the transform.
The trick is that the whole right-hand side is the gradient of the SCALAR
`x -> ljac(x) + dot(g_y, y(x))` with `g_y` frozen at its value at the current `y`
— one reverse pass, not a full Jacobian. Freezing `g_y` is exactly what makes
that identity hold: it is a constant of the differentiation, not a function of
`x_`.

Because the objective is built by applying the block's own `args` closures to the
AD-traced `x_` (see [`Reparametrization`](@ref) and `reparam_rargs`), a location
or log-scale that depends on other parameters is differentiated through as well —
the chain-rule term through `∂args/∂x` is picked up automatically, and no
accessor has to be told about it. A closure that captures a value instead of
reading it from `x_` is a genuine constant and contributes nothing, which is the
intended meaning of a fixed centering.

Cost per gradient evaluation: one inner `logdensity_and_gradient`, one extra
forward evaluation of the transform, and one AD pass over it. This is the
gradient hot path, so the accessor closures inside each `Reparametrization` have
to be AD-friendly.
"""
function _logdensity_and_gradient_reparam(p::ReparametrizedProblem, x::AbstractVector)
    ljac, y = p.reparametrizer(x)
    ld, g_y = LogDensityProblems.logdensity_and_gradient(p.problem, y)
    function reparam_objective(x_)
        ljac_, y_ = p.reparametrizer(x_)
        ljac_ + dot(g_y, y_)
    end
    _, g_x = value_and_gradient(reparam_objective, p.ad_backend, x)
    ljac + ld, g_x
end

# --- Abstract reparametrization interface ---

abstract type AbstractReparametrization end
(t::AbstractReparametrization)(x) = with_logabsdet_jacobian!(copy(x), t, x)
with_logabsdet_jacobian!(Y::AbstractMatrix, t, X::AbstractMatrix) = begin
    map(eachcol(Y), eachcol(X)) do y, x
        with_logabsdet_jacobian!(y, t, x)[1]
    end, Y
end

"""
    PartiallyCentered(c)

One coordinate's centering, `c ∈ [0, 1]`: `c = 1` is fully centered, `c = 0`
fully non-centered, and anything in between is a partial centering.

For a coordinate with location `loc` and log-scale `log_scale`, mapping a value
`x` from `PartiallyCentered(source)` to `PartiallyCentered(target)` is

    y    = target * loc + (x - source * loc) * exp(log_scale * (target - source))
    ljac = log_scale * (target - source)

so `source == target` is the identity, and `source = 1, target = 0` is exactly
the textbook non-centering — subtract the location, divide by the scale.

Use `Float64` centerings (`PartiallyCentered(1.0)`, not `PartiallyCentered(1)`):
warm-up writes the fitted value back into the same `pairs` vector it read, and an
`Int`-parameterized element cannot hold a `Float64` centering. This fails *late* —
not at construction, but with `MethodError: Cannot convert` at the end of the
first restarting warm-up window, the first time a centering is written back.

# How the value gets chosen

At each warm-up window that restarts, every reparametrized coordinate is scored
against a fixed grid of 11 candidate centerings, `range(0, 1, 11)`. Each
candidate accumulates the correlation between the candidate-coordinate position
and its gradient over the recorded halo states, and the candidate minimizing that
correlation becomes the coordinate's new `source`. A perfectly conditioned
coordinate has position and gradient exactly anti-correlated, so the minimum is
the most standard-normal-looking candidate. Coordinates that saw 2 or fewer halo
states keep the centering they had.

The grid is deliberate, not an approximation of a continuous search. Centering is
a bounded one-dimensional quantity, so the candidate set can be *enumerated*; and
because it is enumerable, all candidates can be scored in a single pass over the
halo with online accumulators — every state is visited once, per coordinate,
regardless of how many candidates there are, and nothing has to be stored or
re-visited for an inner optimization loop. Resolution finer than `0.1` is not
what decides how the sampler behaves here. The grid size is fixed; no sampler
keyword exposes it.

See [`Reparametrization`](@ref) for the object that pairs a `target` with an
adapted `source`.
"""
struct PartiallyCentered{C}
    c::C
end
reparam(target::PartiallyCentered, source::PartiallyCentered, x::Real, loc::Real, log_scale::Real) = begin
    ljac = log_scale * (target.c - source.c)
    ljac, target.c * loc + xexpy(x - source.c * loc, ljac)
end
reparam(target::PartiallyCentered, source::PartiallyCentered, x::Real, g::Real, loc::Real, log_scale::Real) = begin
    ljac = log_scale * (target.c - source.c)
    s = exp(ljac)
    ljac, target.c * loc + (x - source.c * loc) * s, g / s
end

"""
    Reparametrization(target, source, args...)

One coordinate's rule: map that coordinate's value from `source` coordinates into
`target` coordinates. Both are [`PartiallyCentered`](@ref).

* `target` is the parametrization the wrapped problem is written in. It is FIXED
  — warm-up never touches it.
* `source` is the parametrization the sampler works in. Warm-up REPLACES it at
  every restarting window (see [`PartiallyCentered`](@ref)).

Construct both with the same centering to start the sampler in the model's own
parametrization and let warm-up move it from there.

`args` are the extra arguments the centerings need — for [`PartiallyCentered`](@ref)
exactly two, the location and the log-scale, in that order. Each is either a
constant or a callable applied to the whole parameter vector, so `x -> x[9]`
reads the location off coordinate 9 and `0.` pins it to zero.

The callables are applied to the sampler's SOURCE-coordinate vector as it was on
entry: [`IndexedReparametrization`](@ref) writes its output into a copy, so no
coordinate's rule ever observes another coordinate's transformed value. They also
run under AD on every gradient evaluation (see [`ReparametrizedProblem`](@ref)),
so keep them cheap and type-generic — index, arithmetic, `exp`/`log`, not
`Float64`-annotated code or anything that mutates.

`InverseFunctions.inverse` returns the same rule with `target` and `source`
swapped.

# Example
```julia
# Eight schools: the group effects have their location at coordinate 9 and their
# log-scale at coordinate 10. This is the rule for ONE group effect — see
# `IndexedReparametrization` for attaching it to coordinates 1:8.
Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0), x -> x[9], x -> x[10])
```
"""
struct Reparametrization{T,S,A}
    target::T
    source::S
    args::A
    Reparametrization(target, source, args...) = new{typeof(target), typeof(source), typeof(args)}(target, source, args)
end
InverseFunctions.inverse((;target, source, args)::Reparametrization) = Reparametrization(source, target, args...)
reparam_rargs((;source, args)::Reparametrization, xi, x) = (source, xi, map(Base.Fix2(maybecall, x), args)...)
reparam_rargs((;source, args)::Reparametrization, xi, gi, x) = (source, xi, gi, map(Base.Fix2(maybecall, x), args)...)
reparam(r::Reparametrization, xi, x) = reparam(r.target, reparam_rargs(r, xi, x)...)
reparam(r::Reparametrization, xi, gi, x) = reparam(r.target, reparam_rargs(r, xi, gi, x)...)

"""
    IndexedReparametrization(pairs)

The container [`ReparametrizedProblem`](@ref) takes: a vector of
`idx => Reparametrization(...)` pairs. Coordinates not listed pass through
unchanged, and the log-Jacobians of the listed ones are summed.

`idx` is a single, RAW index into the unconstrained parameter vector — one pair
per scalar coordinate, not per model parameter. Mapping "the 8 group effects" or
"`sigma`" onto integer positions is the caller's job, and it is the part that
actually costs effort in practice; `web/src/posteriordb_reparametrizations.jl`
in this repo does it for a handful of PosteriorDB models and is worth copying
from.

An empty `IndexedReparametrization` is a no-op. That is the default for every
problem the sampler does not recognise, so an empty one silently samples exactly
as if there were no reparametrization at all.

# Example
```julia
# Eight schools: coordinates 1:8 are the group effects, sharing a location at
# coordinate 9 and a log-scale at coordinate 10.
ir = IndexedReparametrization(
    1:8 .=> Ref(Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(1.0),
        x -> x[9], x -> x[10]
    ))
)
```

# Mutation, ordering and checkpoints

`pairs` is mutated IN PLACE by warm-up: at every restarting window each entry is
replaced by one carrying the newly fitted `source` centering. Two consequences:

* **One reparametrized problem per chain.** Every multi-chain sampler gives chain
  `i` its own `deepcopy` of the `lpdf` you pass, so no two chains ever write the
  same `IndexedReparametrization` and the object you built is not mutated. (Of the
  three, `adaptive_warmup_mcmc` and `cooperative_warmup_mcmc` are the ones that
  adapt a reparametrization at all; `clustered_warmup_mcmc` has no hooks for it
  and does not accept `nonlinear_adapt`.) `adaptive_warmup_mcmc`'s
  `lpdfs::AbstractArray` method is used exactly as given, so
  `adaptive_warmup_mcmc(rngs, [make_problem() for _ in rngs])` is the explicit
  spelling of the default and `fill(lpdf, length(rngs))` is how you deliberately
  opt back into sharing one.
* **The order of `pairs` is load-bearing across a checkpoint/resume.** A
  checkpoint stores only the fitted `source` centerings, as a bare positional
  list; on resume they are zipped back onto the freshly supplied problem's
  `pairs` *by position*, and the stored indices are not consulted. If the
  rebuilt problem enumerates its coordinates in a different order — a
  `Dict`-driven build, a data-dependent sort — every centering silently lands on
  the wrong coordinate. A different *length* is not caught gracefully either: it
  throws `DimensionMismatch`, or, when the overlap collapses to one entry,
  silently overwrites every pair with that one. Build `pairs` deterministically,
  in the same order and with the same length, on both sides of a resume.
  Order also defines accessor dependencies during inversion: if one transformed
  coordinate is read by another block's `loc` or `log_scale`, put the provider
  first so its source coordinate is recovered before the dependent block.
* **The vector element type must retain every accessor's concrete type.** Julia
  widens a vector that mixes constants and differently typed closures, and AD
  then fails late inside warm-up. Construction rejects that shape immediately.
  Use a single concretely typed callable representation for accessors stored in
  one vector.
"""
struct IndexedReparametrization{P} <: AbstractReparametrization
    pairs::P
    function IndexedReparametrization(pairs::P) where {P}
        isempty(pairs) || isconcretetype(fieldtype(eltype(pairs), 2)) || throw(ArgumentError(
            "IndexedReparametrization pairs must have a concrete element type; " *
            "avoid mixing constants and differently typed accessor closures in one vector " *
            "(got eltype $(eltype(pairs)), first element type $(typeof(first(pairs))))",
        ))
        new{P}(pairs)
    end
end
with_logabsdet_jacobian!(y::AbstractVector, (;pairs)::IndexedReparametrization, x::AbstractVector) = begin
    ljac = 0.
    for (idx, value) in pairs
        tmp, y[idx] = reparam(value, x[idx], x)
        ljac += tmp
    end
    ljac, y
end

struct InverseIndexedReparametrization{I} <: AbstractReparametrization
    forward::I
end
InverseFunctions.inverse(ir::IndexedReparametrization) = InverseIndexedReparametrization(ir)
InverseFunctions.inverse(ir::InverseIndexedReparametrization) = ir.forward

# Recover the source coordinates of an indexed transform from a point in its
# target coordinates. Each accessor is evaluated on the source point being
# reconstructed. That distinction is essential when a transformed coordinate
# is itself another block's location or scale.
#
# `pairs` order is the dependency order: a block may read an earlier recovered
# coordinate. This is the same deterministic, load-bearing order already used
# by `optimize!` and checkpoint restoration.
function _inverse_with_logabsdet_jacobian!(source::AbstractVector,
                                           ir::IndexedReparametrization,
                                           target::AbstractVector)
    source .= target
    ljac = zero(eltype(target))
    for (idx, value) in ir.pairs
        ljac_i, source_i = reparam(inverse(value), target[idx], source)
        source[idx] = source_i
        ljac += ljac_i
    end
    ljac, source
end
_inverse_with_logabsdet_jacobian(ir::IndexedReparametrization,
                                 target::AbstractVector) =
    _inverse_with_logabsdet_jacobian!(copy(target), ir, target)
with_logabsdet_jacobian!(source::AbstractVector,
                         (;forward)::InverseIndexedReparametrization,
                         target::AbstractVector) =
    _inverse_with_logabsdet_jacobian!(source, forward, target)

# --- Online reparametrization loss tracking ---

struct OnlineReparametrizationLoss{L<:OnlineStatsBase.Mean,C<:OnlineStatsBase.CovMatrix}
    ljac::L
    cov::C
end
OnlineStatsBase.nobs((;ljac)::OnlineReparametrizationLoss) = OnlineStatsBase.nobs(ljac)
OnlineReparametrizationLoss(::AbstractMatrix) = OnlineReparametrizationLoss()
OnlineReparametrizationLoss(::AbstractMatrix, ::AbstractMatrix) = OnlineReparametrizationLoss()
OnlineReparametrizationLoss() = OnlineReparametrizationLoss(OnlineStatsBase.Mean(), OnlineStatsBase.CovMatrix())
OnlineStatsBase.fit!((;ljac, cov)::OnlineReparametrizationLoss, obs) = map(OnlineStatsBase.fit!, (ljac, cov), (obs[1], [obs[2], obs[3]]))
reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss; w1=0, w2=1-w1) = (
    w1 * (-mean(ljac) + .5 * log(Statistics.cov(cov)[1, 1])) + w2 * Statistics.cor(cov)[1, 2]
)
scale_estimate(orl::OnlineReparametrizationLoss) = begin
    c = Statistics.cov(orl.cov)
    (c[1,1] / c[2,2])^.25
end

# --- OnlineReparametrizer: fits multiple candidates ---

struct OnlineReparametrizer{P}
    pairs::P
end
OnlineStatsBase.fit!((;pairs)::OnlineReparametrizer, args...) = for (candidate, accumulator) in pairs
    OnlineStatsBase.fit!(accumulator, reparam(candidate, args...))
end
OnlineStatsBase.nobs((;pairs)::OnlineReparametrizer) = length(pairs) == 0 ? 0 : OnlineStatsBase.nobs(pairs[1][2])
minimizer((;pairs)::OnlineReparametrizer; kwargs...) = argmin(p -> reparametrization_loss(last(p); kwargs...), pairs)
scale_estimate(or::OnlineReparametrizer; kwargs...) = scale_estimate(last(minimizer(or; kwargs...)))

reparametrization_candidates(::PartiallyCentered; n=11) = Iterators.map(PartiallyCentered, range(0, 1, n))
OnlineReparametrizer((;source)::Reparametrization, xg...; kwargs...) = OnlineReparametrizer(source, xg...; kwargs...)
OnlineReparametrizer(source::PartiallyCentered, xg...; kwargs...) = OnlineReparametrizer([
    target => OnlineReparametrizationLoss(xg...)
    for target in reparametrization_candidates(source; kwargs...)
])

# --- Batch optimization ---

optimize!((;pairs)::IndexedReparametrization, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    pairs .= Base.broadcasted(pairs) do (idx, value)
        or = OnlineReparametrizer(value, xg...; kwargs...)
        for xgi in zip(eachcol.(xg)...)
            OnlineStatsBase.fit!(or, reparam_rargs(value, getindex.(xgi, idx)..., first(xgi))...)
        end
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_value = Reparametrization(value.target, first(minimizer(or; loss_kwargs...)), value.args...)
        trans = Reparametrization(new_value.source, value.source, value.args...)
        for xgi in zip(eachcol.(xg)...)
            setindex!.(xgi, reparam(trans, getindex.(xgi, idx)..., first(xgi))[2:end], idx)
        end
        idx => new_value
    end
    IndexedReparametrization(pairs)
end

# --- Online fitting for IndexedReparametrization ---

OnlineReparametrizer((;pairs)::IndexedReparametrization; kwargs...) = OnlineReparametrizer([
    idx => OnlineReparametrizer(value; kwargs...)
    for (idx, value) in pairs
])
OnlineStatsBase.fit!(ir::IndexedReparametrization, ors::OnlineReparametrizer, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    ir.pairs .= Base.broadcasted(ir.pairs, ors.pairs) do (idx, value), (_, or)
        for xgi in zip(eachcol.(xg)...)
            OnlineStatsBase.fit!(or, reparam_rargs(value, getindex.(xgi, idx)..., first(xgi))...)
        end
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_value = Reparametrization(value.target, first(minimizer(or; loss_kwargs...)), value.args...)
        trans = Reparametrization(new_value.source, value.source, value.args...)
        for xgi in zip(eachcol.(xg)...)
            setindex!.(xgi, reparam(trans, getindex.(xgi, idx)..., first(xgi))[2:end], idx)
        end
        idx => new_value
    end
    ir
end

# --- Hooks for adaptive_warmup_mcmc ---

_reparametrization_ad_backend(p::ReparametrizedProblem) = p.ad_backend
_reparametrization_ad_backend(p::WrappedLogDensityProblem) =
    _reparametrization_ad_backend(parent(p))

struct ReparametrizationTransportObjective{N,O,G}
    new_ir::N
    old_ir::O
    old_gradient::G
end

function (objective::ReparametrizationTransportObjective)(new_position)
    ljac_new, model_position = objective.new_ir(new_position)
    ljac_old, old_position =
        _inverse_with_logabsdet_jacobian(objective.old_ir, model_position)
    ljac_new + ljac_old + dot(objective.old_gradient, old_position)
end

"""
    _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                             position, gradient)

Transport halo positions and gradients from `old_ir`'s source coordinates to
the final source coordinates selected in `reparametrizer(lpdf)`. The marginal
search in `optimize!` may mutate its working pool as it selects each source;
this final pass rematerializes the pool from the original positions and
gradients once every source has settled.

For the new-to-old map `S = inverse(old_ir) ∘ new_ir`, the transported gradient
is

```
∇ log|J_S(x_new)| + J_S(x_new)' * g_old.
```

Compute the two terms together as the gradient of the scalar
`log|J_S(x_new)| + dot(g_old, S(x_new))`. This is one AD pass through the full
transform per halo column, so dependencies of a block's location or scale on
other coordinates contribute their chain-rule terms. The wrapped model is
never evaluated.
"""
function _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                                  position, gradient)
    new_ir = reparametrizer(lpdf)
    backend = _reparametrization_ad_backend(lpdf)
    columns = zip(eachcol(old_position), eachcol(old_gradient),
                  eachcol(position), eachcol(gradient))
    for (x_old, g_old, x_new, g_new) in columns
        _, y = old_ir(x_old)
        _, transported_position = _inverse_with_logabsdet_jacobian(new_ir, y)
        x_new .= transported_position
        transport_objective = ReparametrizationTransportObjective(
            new_ir, old_ir, collect(g_old),
        )
        _, transported_gradient = value_and_gradient(transport_objective, backend, x_new)
        g_new .= transported_gradient
    end
    gradient
end

find_reparametrization!(lpdf, halo_position, halo_gradient, position_and_gradient) = begin
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return position_and_gradient
    old_ir = IndexedReparametrization(copy(ir.pairs))
    old_position = copy(halo_position)
    old_gradient = copy(halo_gradient)
    optimize!(ir, halo_position, halo_gradient)
    _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                             halo_position, halo_gradient)
    DynamicHMC.evaluate_ℓ(lpdf, position_and_gradient.q; strict=false)
end

# Draws are stored in the SAMPLING parametrization: `logdensity` receives the
# sampler position in `source` coordinates and maps it through `ir`
# (source -> target) before handing `y` to the inner problem. Reporting draws in
# the model's own (`target`) parametrization therefore applies `ir` itself.
# Applying `inverse(ir)` here maps target -> source, i.e. the wrong direction —
# it was a no-op only while adaptation left `source == target`.
reparametrize!(lpdf, posterior_position) = begin
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return
    for col in eachcol(posterior_position)
        ljac, y = ir(col)
        col .= y
    end
end
