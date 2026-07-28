# Generic adaptive nonlinear reparametrization machinery
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

maybecall(f::Function, args...; kwargs...) = f(args...; kwargs...)
maybecall(x, args...; kwargs...) = x

struct DirectCandidateScoring end
const DIRECT_CANDIDATE_SCORING = DirectCandidateScoring()

"""
    CandidateScoringPlan(prepare, score; synchronize! = identity)

A pluggable strategy for scoring reparametrization candidates from online
evidence, attached with
`ReparametrizedProblem(ir, problem, backend; scoring_plan = plan)`.

Every `ReparametrizedProblem` has a plan — the public `nothing` default resolves
to an internal direct-scoring plan. Passing one therefore replaces a strategy
rather than enabling one. Declining a pair is not the same as having no plan:
`score` returning `nothing` restores the direct scoring formula for that pair,
but attaching any custom plan already determines the evidence source, so a plan
that declines every pair is still not equivalent to the default.

# Callbacks

`prepare(ir, position, gradient)` runs once per online evidence observation and
returns a transient frame. The frame is opaque to WarmupHMC and is passed
unchanged to every `score` call arising from that observation, so it is where
work shared across candidates belongs rather than being repeated per pair.

`score(frame, pair_number, idx, reparametrization, candidate)` returns
`(ljac, position, gradient)` for `candidate`, where `ljac` is the log-Jacobian
term in `logdensity(rp, x) == ljac + logdensity(problem, y)`. `pair_number` is
the pair's one-based position in `ir.pairs`; `idx` is that pair's source-vector
coordinate selector. Returning `nothing` instead delegates that one pair to
direct scoring; the choice is per pair, so a plan may score only the pairs it
has an opinion about.

`synchronize!(ir)` runs after construction, after every winner commit, and when
a checkpoint source is restored — the points at which `ir` may have changed
underneath a plan holding derived state. The default `identity` is correct for
a stateless plan.

# Evidence source

A custom plan interprets `nonlinear_evidence=:linear_pool` as online
all-good-leaf evidence; it never replays the retained linear pool.

# Checkpoints

Checkpoint payloads are `Serialization`-based and deliberately exclude the
log-density and the scoring plan, so a plan does not survive a checkpoint. The
marker is compared on restore and a mismatch is an error in either direction:
restoring a plan-recorded run without attaching one, and restoring a
default-recorded run with a plan attached, both fail loudly rather than falling
back. A silent fallback would resume a different problem than the one recorded
while looking like a successful restore.

See [`ReparametrizedProblem`](@ref) for the wrapper this attaches to, and
[Adaptive centering at fixed `c`](@ref) for a measured comparison of a
strict-online scoring proxy against an exact-score reference.
"""
struct CandidateScoringPlan{P,S,Y}
    prepare::P
    score::S
    synchronize!::Y
end
CandidateScoringPlan(prepare, score; synchronize! = identity) =
    CandidateScoringPlan(prepare, score, synchronize!)

_synchronize_scoring!(::DirectCandidateScoring, ir) = ir
_synchronize_scoring!(plan::CandidateScoringPlan, ir) =
    (plan.synchronize!(ir); ir)

"""
    ReparametrizedProblem(reparametrizer, problem, ad_backend=nothing; scoring_plan=nothing)

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
operation count, and an operation count is not wall-clock. The objective
differentiated here is scalar in the *full* parameter vector, so forward mode
costs `ceil(n / chunksize)` sweeps of the transform per gradient while reverse
mode costs one.

!!! warning "That argument is an operation count, and how it scales here is untested"
    It is tempting to conclude that the gap widens with dimension — exactly where
    reparametrization is worth doing. That has **not** been established.

    **This docstring deliberately carries no benchmark figures.** It used to, and
    they went stale three times: a point estimate that no aggregation reproduced,
    then a range that spanned nothing in the repository, then a range correctly
    derived from every checked-in source that a re-measurement invalidated within
    minutes. Each fix addressed how the number was *chosen*; none addressed the
    actual cause, which is that a docstring is a string evaluated at package load
    and so cannot read the results JSON the way the manual's tables do. It is the
    one surface here with no build-time seam, so any figure typed into it is
    dated silently by the next regeneration and nothing in the build can tell.

    For every current number — per-target Enzyme-over-ForwardDiff ratios, the
    `Duplicated` cost, and the boxed/unboxed A/B — read
    [What the backend costs, measured](@ref). Those tables are generated from
    `docs/benchmark/results/` at build time, so they move when the data moves.

    What is stable enough to state here, and why:

      * **Reverse mode wins on every clean measurement to date.** A direction,
        not a magnitude; it has replicated across every harness in the repo.
      * **The boxing confound is fixed.** The larger targets — `seeds`,
        `radon_partially_pooled`, `radon_variable_intercept` — were once measured
        against specs whose accessor closures captured a `Core.Box` (fixed in
        `e9bcfd0`; see `_index_getter` in
        `web/src/posteriordb_reparametrizations.jl`). Boxing was perfectly
        correlated with the apparent ranking — five for five, every boxed spec
        one where Enzyme lost — so those rows measured the defect, not the
        backend. The probe now reports no boxed spec at all.
      * **Scaling with dimension is still unresolved**, but no longer because of
        that confound. On the current data, target identity accounts for
        essentially all of the spread and `d` for very little, so these targets
        do not settle a slope either way.
      * **Sampling is unaffected.** ESS per 1000 gradients, gradient counts, the
        fitted `c` and stuck-adaptation counts are identical across backends; the
        backend sets the cost of a gradient, not how many are needed.
      * **DifferentiationInterface re-prepares on every call**, and reusing a
        prep object measured neutral-to-worse. That conclusion has replicated;
        the share of the call it represents has not, so it is not quoted here.
      * Running a sampler before timing warms the ForwardDiff path enough to
        flatter it in a naive microbenchmark.

    So: pick the backend by measuring your own target. This docstring has twice
    claimed a universal default and been wrong both times, most recently by
    publishing the boxing artifact as a property of the backend.

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
    **Do not take that hint.** It is correct — it agrees with `Const` to within
    floating-point noise — but it allocates and propagates a shadow copy of the
    closure on every call. The hint diagnoses the problem; it is not the fix.

    How much that costs is **target-dependent** and is not quoted here, for the
    reason given in the previous warning: earlier revisions published first a
    single ratio and then a two-row table, and both were dated by the next
    regeneration. The current per-target figures are in the generated table under
    [What the backend costs, measured](@ref). The funnel is the extreme case
    rather than a representative one, which is why no single number belongs here
    in any revision.

    One reading to avoid: the larger targets once measured ≈1.0×, and that was
    taken to mean `Const` only helps on small targets. It was an artifact of the
    boxed specs, where the boxing dominated the call — it says boxing cost more
    than the shadow copy, not that the shadow copy is cheap at high dimension.
    That framing does not survive the fix, and the de-boxed table no longer shows
    a row where `Duplicated` is free.

    None of that changes the recommendation. `Const` is the right annotation on
    **correctness** grounds everywhere — `g_y` is frozen by construction — and it
    is never slower.

A backend is required in practice, and omitting it fails *late*: the
two-argument constructor `ReparametrizedProblem(r, p)` stores `nothing`, which
is not a backend, so `logdensity` keeps working on that object and the first
`logdensity_and_gradient` call `MethodError`s inside `value_and_gradient`.
Construction itself never complains.

# Candidate scoring

`scoring_plan` attaches a [`CandidateScoringPlan`](@ref), which replaces how a
candidate reparametrization is scored during warm-up. The default is `nothing`
and takes the original path — not an equivalent one, the same one — so attaching
no plan cannot move a result.

A plan supplies `prepare`, `score` and `synchronize!`. `prepare(ir, position,
gradient)` runs once per online evidence observation and returns a transient
frame; `score(frame, pair_number, idx, value, candidate)` returns `(ljac,
position, gradient)`, or `nothing` to fall through to the existing direct
scoring. `synchronize!(ir)` runs after construction, after a winner is
committed, and after a checkpoint restore — before anything transports or
evaluates the reparametrization.

The score is a **fixed-frame proxy**: the frame is held fixed while a candidate
is evaluated, so it does not model every coordinate moving at once. That is a
deliberate approximation, not an oversight, and it is the reason a plan is opt-in
rather than the default.

!!! warning "A plan does not survive a checkpoint — re-attach it on restore"
    A plan is three functions, and checkpoint payloads deliberately exclude the
    supplied log-density problem, so the plan is not serialized with them. What a
    checkpoint records is *that* a non-default plan was in force. Restore then
    fails loudly if that marker and the supplied plan disagree in either
    direction, rather than falling back to the default. The failure is deliberate:
    a resumed run that quietly scored differently from the run it resumed would
    diverge with nothing in the output saying so.

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
struct ReparametrizedProblem{R,P,B,S}
    reparametrizer::R
    problem::P
    ad_backend::B
    scoring_plan::S
    function ReparametrizedProblem(r::R, p::P, b::B, scoring::S) where {R,P,B,S}
        _synchronize_scoring!(scoring, r)
        new{R,P,B,S}(r, p, b, scoring)
    end
end
_resolve_candidate_scoring(::Nothing) = DIRECT_CANDIDATE_SCORING
_resolve_candidate_scoring(plan::CandidateScoringPlan) = plan
ReparametrizedProblem(r, p, b; scoring_plan=nothing) =
    ReparametrizedProblem(r, p, b, _resolve_candidate_scoring(scoring_plan))
ReparametrizedProblem(r, p; scoring_plan=nothing) =
    ReparametrizedProblem(r, p, nothing; scoring_plan)
reparametrizer(p::ReparametrizedProblem) = p.reparametrizer
reparametrizer(p::WrappedLogDensityProblem) = reparametrizer(parent(p))
reparametrizer(::Any) = IndexedReparametrization([])
candidate_scoring_plan(p::ReparametrizedProblem) = p.scoring_plan
candidate_scoring_plan(p::WrappedLogDensityProblem) = candidate_scoring_plan(parent(p))
candidate_scoring_plan(::Any) = DIRECT_CANDIDATE_SCORING
_has_custom_candidate_scoring(p) = candidate_scoring_plan(p) isa CandidateScoringPlan
_synchronize_scoring!(p) = _synchronize_scoring!(candidate_scoring_plan(p), reparametrizer(p))
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
constant or a `Function` applied to the whole parameter vector, so `x -> x[9]`
reads the location off coordinate 9 and `0.` pins it to zero. A callable struct
must subtype `Function`; an otherwise-callable object is treated as a constant by
the accessor contract. That mismatch is silent at construction: the struct
itself becomes the argument value, so any symptom appears later inside the
centering or gradient evaluation.

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
OnlineReparametrizationLoss() = OnlineReparametrizationLoss(
    OnlineStatsBase.Mean(), OnlineStatsBase.CovMatrix(),
)
function OnlineStatsBase.fit!(loss::OnlineReparametrizationLoss, obs;
                              weight::Real=1, count::Bool=true)
    weight == 1 && count || throw(ArgumentError(
        "weighted updates require WeightedReparametrizationLoss",
    ))
    map(OnlineStatsBase.fit!, (loss.ljac, loss.cov), (obs[1], [obs[2], obs[3]]))
end
reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss; w1=0, w2=1-w1) = (
    w1 * (-mean(ljac) + .5 * log(Statistics.cov(cov)[1, 1])) +
    w2 * Statistics.cor(cov)[1, 2]
)
scale_estimate(loss::OnlineReparametrizationLoss) = begin
    c = Statistics.cov(loss.cov)
    (c[1, 1] / c[2, 2])^.25
end

mutable struct WeightedReparametrizationLoss
    weight::Float64
    weight2::Float64
    mean_ljac::Float64
    mean_position::Float64
    mean_gradient::Float64
    m2_position::Float64
    m2_gradient::Float64
    co_position_gradient::Float64
    groups::Int
end
OnlineStatsBase.nobs(loss::WeightedReparametrizationLoss) = loss.groups
WeightedReparametrizationLoss(::AbstractMatrix) = WeightedReparametrizationLoss()
WeightedReparametrizationLoss(::AbstractMatrix, ::AbstractMatrix) = WeightedReparametrizationLoss()
WeightedReparametrizationLoss() = WeightedReparametrizationLoss(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
)

function OnlineStatsBase.fit!(loss::WeightedReparametrizationLoss, obs;
                              weight::Real=1, count::Bool=true)
    weight >= 0 || throw(ArgumentError("nonlinear observation weight must be nonnegative, got $weight"))
    iszero(weight) && return loss
    w = Float64(weight)
    ljac, position, gradient = obs
    new_weight = loss.weight + w
    fraction = w / new_weight

    delta_ljac = ljac - loss.mean_ljac
    delta_position = position - loss.mean_position
    delta_gradient = gradient - loss.mean_gradient
    new_mean_position = loss.mean_position + fraction * delta_position
    new_mean_gradient = loss.mean_gradient + fraction * delta_gradient

    loss.mean_ljac += fraction * delta_ljac
    loss.mean_position = new_mean_position
    loss.mean_gradient = new_mean_gradient
    loss.m2_position += w * delta_position * (position - new_mean_position)
    loss.m2_gradient += w * delta_gradient * (gradient - new_mean_gradient)
    loss.co_position_gradient += w * delta_position * (gradient - new_mean_gradient)
    loss.weight = new_weight
    loss.weight2 += abs2(w)
    count && (loss.groups += 1)
    loss
end

function reset!(loss::WeightedReparametrizationLoss)
    loss.weight = 0
    loss.weight2 = 0
    loss.mean_ljac = 0
    loss.mean_position = 0
    loss.mean_gradient = 0
    loss.m2_position = 0
    loss.m2_gradient = 0
    loss.co_position_gradient = 0
    loss.groups = 0
    loss
end

function reparametrization_loss(loss::WeightedReparametrizationLoss; w1=0, w2=1-w1)
    covariance_denom = loss.weight - loss.weight2 / loss.weight
    variance_position = loss.m2_position / covariance_denom
    correlation = loss.co_position_gradient /
        sqrt(loss.m2_position * loss.m2_gradient)
    w1 * (-loss.mean_ljac + .5 * log(variance_position)) + w2 * correlation
end
scale_estimate(loss::WeightedReparametrizationLoss) = begin
    (loss.m2_position / loss.m2_gradient)^.25
end
effective_n(loss::WeightedReparametrizationLoss) = loss.weight^2 / loss.weight2

# --- OnlineReparametrizer: fits multiple candidates ---

struct OnlineReparametrizer{P}
    pairs::P
end
OnlineStatsBase.fit!((;pairs)::OnlineReparametrizer, args...; kwargs...) = for (candidate, accumulator) in pairs
    OnlineStatsBase.fit!(accumulator, reparam(candidate, args...); kwargs...)
end
OnlineStatsBase.nobs((;pairs)::OnlineReparametrizer) = length(pairs) == 0 ? 0 : OnlineStatsBase.nobs(pairs[1][2])
minimizer((;pairs)::OnlineReparametrizer; kwargs...) = argmin(p -> reparametrization_loss(last(p); kwargs...), pairs)
scale_estimate(or::OnlineReparametrizer; kwargs...) = scale_estimate(last(minimizer(or; kwargs...)))
reset!((;pairs)::OnlineReparametrizer) = (foreach(p -> reset!(last(p)), pairs); nothing)
_mark_group!(loss::WeightedReparametrizationLoss) = (loss.groups += 1; loss)
_mark_group!((;pairs)::OnlineReparametrizer) = foreach(p -> _mark_group!(last(p)), pairs)

reparametrization_candidates(::PartiallyCentered; n=11) = Iterators.map(PartiallyCentered, range(0, 1, n))
OnlineReparametrizer((;source)::Reparametrization, xg...; kwargs...) = OnlineReparametrizer(source, xg...; kwargs...)
OnlineReparametrizer(source::PartiallyCentered, xg...;
                     accumulator=OnlineReparametrizationLoss, kwargs...) = OnlineReparametrizer([
    target => accumulator(xg...)
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
function _fit_candidate_scoring!(::DirectCandidateScoring,
                                 ir::IndexedReparametrization,
                                 ors::OnlineReparametrizer,
                                 position::AbstractVector,
                                 gradient::AbstractVector;
                                 weight::Real=1, count::Bool=true)
    for ((idx, value), (stored_idx, or)) in zip(ir.pairs, ors.pairs)
        idx == stored_idx || throw(ArgumentError(
            "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
        ))
        OnlineStatsBase.fit!(
            or,
            reparam_rargs(value, position[idx], gradient[idx], position)...;
            weight, count,
        )
    end
    ir
end

function _fit_candidate_scoring!(plan::CandidateScoringPlan,
                                 ir::IndexedReparametrization,
                                 ors::OnlineReparametrizer,
                                 position::AbstractVector,
                                 gradient::AbstractVector;
                                 weight::Real=1, count::Bool=true)
    frame = plan.prepare(ir, position, gradient)
    for (pair_number, ((idx, value), (stored_idx, or))) in
        enumerate(zip(ir.pairs, ors.pairs))
        idx == stored_idx || throw(ArgumentError(
            "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
        ))
        for (candidate, accumulator) in or.pairs
            observation = plan.score(frame, pair_number, idx, value, candidate)
            if isnothing(observation)
                observation = reparam(
                    candidate,
                    reparam_rargs(value, position[idx], gradient[idx], position)...,
                )
            end
            OnlineStatsBase.fit!(accumulator, observation; weight, count)
        end
    end
    ir
end

function OnlineStatsBase.fit!(ir::IndexedReparametrization, ors::OnlineReparametrizer,
                              position::AbstractVector, gradient::AbstractVector;
                              weight::Real=1, count::Bool=true,
                              scoring_plan=DIRECT_CANDIDATE_SCORING)
    _fit_candidate_scoring!(
        scoring_plan, ir, ors, position, gradient; weight, count,
    )
end

function optimize!(ir::IndexedReparametrization, ors::OnlineReparametrizer;
                   loss_kwargs=(;))
    ir.pairs .= Base.broadcasted(ir.pairs, ors.pairs) do (idx, value), (stored_idx, or)
        idx == stored_idx || throw(ArgumentError(
            "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
        ))
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_source = first(minimizer(or; loss_kwargs...))
        idx => Reparametrization(value.target, new_source, value.args...)
    end
    ir
end

const NONLINEAR_EVIDENCE_MODES = (:linear_pool, :all_good_leaves, :nuts_weighted)
const NONLINEAR_TRAJECTORY_WEIGHTINGS = (
    :unit,
    :stepsize,
)

mutable struct NonlinearRecorder{O,T}
    mode::Symbol
    trajectory_weighting::Symbol
    good_leaf_threshold::T
    online::O
end

function NonlinearRecorder(lpdf; mode=:linear_pool, trajectory_weighting=:unit,
                           good_leaf_threshold=log(1e-2))
    mode in NONLINEAR_EVIDENCE_MODES || throw(ArgumentError(
        "unknown nonlinear evidence mode $mode; expected one of $(join(NONLINEAR_EVIDENCE_MODES, ", "))",
    ))
    trajectory_weighting in NONLINEAR_TRAJECTORY_WEIGHTINGS || throw(ArgumentError(
        "unknown nonlinear trajectory weighting $trajectory_weighting; expected one of " *
        join(NONLINEAR_TRAJECTORY_WEIGHTINGS, ", "),
    ))
    NonlinearRecorder(
        mode,
        trajectory_weighting,
        good_leaf_threshold,
        OnlineReparametrizer(
            reparametrizer(lpdf); accumulator=WeightedReparametrizationLoss,
        ),
    )
end

_uses_online_candidate_scoring(::DirectCandidateScoring) = false
_uses_online_candidate_scoring(::CandidateScoringPlan) = true
function _effective_nonlinear_evidence(recorder::NonlinearRecorder, lpdf)
    plan = candidate_scoring_plan(lpdf)
    recorder.mode === :linear_pool && _uses_online_candidate_scoring(plan) ?
        :all_good_leaves : recorder.mode
end

function _trajectory_weight(weighting, stepsize)
    weighting === :unit && return 1.0
    weighting === :stepsize && return stepsize
    throw(ArgumentError("unsupported nonlinear trajectory weighting $weighting"))
end

function record_nonlinear!(recorder::NonlinearRecorder, lpdf, leaves, stepsize)
    mode = _effective_nonlinear_evidence(recorder, lpdf)
    mode === :linear_pool && return recorder
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return recorder
    trajectory_weight = _trajectory_weight(recorder.trajectory_weighting, stepsize)
    iszero(trajectory_weight) && return recorder

    recorded = false
    for i in eachindex(leaves.dH)
        leaf_weight = if mode === :all_good_leaves
            i != 1 && leaves.dH[i] > recorder.good_leaf_threshold ? 1.0 : 0.0
        else
            leaves.weights[i]
        end
        weight = trajectory_weight * leaf_weight
        iszero(weight) && continue
        OnlineStatsBase.fit!(
            ir,
            recorder.online,
            @view(leaves.position[:, i]),
            @view(leaves.gradient[:, i]);
            weight,
            count=false,
            scoring_plan=candidate_scoring_plan(lpdf),
        )
        recorded = true
    end
    recorded && _mark_group!(recorder.online)
    recorder
end

reset!(recorder::NonlinearRecorder) = (reset!(recorder.online); recorder)
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
    _synchronize_scoring!(lpdf)
    _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                             halo_position, halo_gradient)
    DynamicHMC.evaluate_ℓ(lpdf, position_and_gradient.q; strict=false)
end

function find_reparametrization!(lpdf, recorder::NonlinearRecorder,
                                  halo_position, halo_gradient,
                                  position_and_gradient)
    _effective_nonlinear_evidence(recorder, lpdf) === :linear_pool &&
        return find_reparametrization!(
        lpdf, halo_position, halo_gradient, position_and_gradient,
    )
    ir = reparametrizer(lpdf)
    isempty(ir.pairs) && return position_and_gradient
    old_ir = IndexedReparametrization(copy(ir.pairs))
    old_position = copy(halo_position)
    old_gradient = copy(halo_gradient)
    optimize!(ir, recorder.online)
    _synchronize_scoring!(lpdf)
    _jointly_transport_halo!(
        lpdf, old_ir, old_position, old_gradient, halo_position, halo_gradient,
    )
    reset!(recorder)
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
