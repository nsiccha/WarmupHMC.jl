# Generic adaptive nonlinear reparametrization machinery
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

# A `Reparametrization` argument is a constant number or a callable applied to the
# parameter vector: a `Function` or any callable struct.
maybecall(x::Number, args...; kwargs...) = x
maybecall(f, args...; kwargs...) = f(args...; kwargs...)

# What `IndexedReparametrization` accepts as an accessor at construction. The scalar
# transform takes `Real` locations and log-scales, so a constant that is not `Real`
# (or a value that cannot be called) would only fail later, inside the centering or
# its gradient (snag built-an-indexed-3fed939c).
_is_accessor(a) = a isa Real || a isa Function || !isempty(methods(a))

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

`synchronize!(ir)` runs DURING construction — inside `ReparametrizedProblem`'s
inner constructor, ahead of `new`, so the problem does not exist yet and `ir` is
all a plan can reach — then after every winner commit, and when a checkpoint
source is restored. Those are the points at which `ir` may have changed
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
[`candidate_scoring_losses`](@ref) for the criterion and the supported ways to
inspect its loss curve.
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
    WindowSelectionPlan(select!; synchronize! = identity)

A pluggable rule that CHOOSES a reparametrizer's controls at every restarting
warm-up window, attached with
`ReparametrizedProblem(ir, problem, backend; scoring_plan = plan)`.

It replaces the candidate grid and its position-gradient criterion entirely.
WarmupHMC keeps the rest of the window lifecycle: collecting evidence,
transporting the retained pool and the active point into the newly selected
coordinates, the metric update that follows, and checkpointing. Unlike a
[`CandidateScoringPlan`](@ref), a selection plan also works with reparametrizers
that are not an [`IndexedReparametrization`](@ref) (see *Custom
reparametrizers* below).

# Callbacks

`select!(ir, positions, gradients)` runs at a restarting window boundary. The
columns of `positions` and `gradients` are equally weighted sampler states in
the CURRENT source coordinates of `ir` (read-only; see *Evidence*). It may
update `ir`'s controls in place and returns `true` if it did, in which case the
evidence and the active point are transported into the new coordinates, or
`false` to keep the current ones.

`synchronize!(ir)` has the same role and call points as for a
[`CandidateScoringPlan`](@ref).

# Evidence

With `nonlinear_evidence = :linear_pool` (the default) the evidence is the
retained pool itself (at most `recording_target` states). With
`:all_good_leaves` or `:nuts_weighted` it is `recording_target` states drawn
with replacement from every NUTS leaf of the window, each leaf with probability
proportional to the weight WarmupHMC's own scoring gives it
(`nonlinear_trajectory_weighting` included), so repeated columns are expected.
Unlike a scoring plan, a selection plan never remaps `:linear_pool`.

# Custom reparametrizers

Besides `IndexedReparametrization`, any `WarmupHMC.AbstractReparametrization`
can be adapted by a selection plan if it implements:

* `WarmupHMC.with_logabsdet_jacobian!(y, ir, x) -> (ljac, y)`, the source-to-target
  map and its log-Jacobian (this runs under AD on every gradient evaluation);
* `InverseFunctions.inverse(ir)`, an `AbstractReparametrization` computing the
  target-to-source map in the same way;
* `WarmupHMC.reparam_controls(ir)`, a serializable COPY of the current controls,
  and `WarmupHMC.restore_reparam_controls!(ir, controls)`, which writes such a
  value back and throws if it cannot belong to `ir`;
* optionally `WarmupHMC.snapshot_reparametrization(ir)`, an independent copy
  with the same controls (default `deepcopy`).

# Checkpoints

The plan itself is not serialized; re-attach it on resume, as for a scoring
plan. The controls are written under the payload's `reparam_sources` key.
"""
struct WindowSelectionPlan{S,Y}
    select!::S
    synchronize!::Y
end
WindowSelectionPlan(select!; synchronize! = identity) =
    WindowSelectionPlan(select!, synchronize!)

_synchronize_scoring!(plan::WindowSelectionPlan, ir) =
    (plan.synchronize!(ir); ir)

"""
    ReparametrizedProblem(reparametrizer, problem, ad_backend=nothing; scoring_plan=nothing)

Wrap a LogDensityProblems target with a coordinate map from sampler coordinates
(source) to the model's coordinates (target). The transformed density includes
the map's log-Jacobian. Supply an [`IndexedReparametrization`](@ref), or a custom
reparametrizer controlled by a [`WindowSelectionPlan`](@ref).

The inner target supplies its own gradient. `ad_backend` differentiates the map
and Jacobian through DifferentiationInterface; load the implementing AD package
as well. For Enzyme, load Enzyme and pass `AutoEnzyme()`. The two-argument
constructor supports log-density evaluation but does not provide a sampling
gradient. An empty indexed map is the identity and has no controls to adapt.

`scoring_plan` accepts a [`CandidateScoringPlan`](@ref) or
[`WindowSelectionPlan`](@ref). The default uses WarmupHMC's candidate rule.
A custom plan is not serialized; reconstruct the same strategy when resuming a
checkpoint. Its presence is checked during restoration.

Adaptive and cooperative sampling return `posterior_position` in the inner
model's coordinates, including when `nonlinear_adapt=false`. Do not transform
those returned positions again. Other working arrays retain sampler coordinates.
See [Reparametrization](@ref) for construction and coordinate conventions.
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
_resolve_candidate_scoring(plan::WindowSelectionPlan) = plan
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
_has_custom_candidate_scoring(p) = !(candidate_scoring_plan(p) isa DirectCandidateScoring)
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

The AD pass is [`_reparam_gradient`](@ref), which an extension may specialize.
"""
function _logdensity_and_gradient_reparam(p::ReparametrizedProblem, x::AbstractVector)
    ljac, y = p.reparametrizer(x)
    ld, g_y = LogDensityProblems.logdensity_and_gradient(p.problem, y)
    ljac + ld, _reparam_gradient(p.ad_backend, p.reparametrizer, x, g_y)
end

"""
    WarmupHMC._reparam_gradient(backend, reparametrizer, x, g_y)

The gradient in `x` of `_reparam_objective(x, reparametrizer, g_y)`, i.e. of
`ljac(x) + dot(g_y, y(x))`.

This method differentiates the whole transform in one pass. With a reverse-mode
`AutoEnzyme` backend and an [`IndexedReparametrization`](@ref), the Enzyme
extension differentiates it pair by pair instead, in reverse pair order, and a
run of pairs that share accessor work (see [`SharedAccessor`](@ref)) in one call.
The result agrees to rounding: a coordinate that several pairs read may sum their
contributions in a different order. The pair loop before runs allocated for every
accessor call in one pass (about 770 bytes per pair on a correlated random-effect
block, snag `find-reparametri-e2a5b2a7`); the run-wise map allocates a few working
vectors in one pass on that block, and pair by pair Enzyme allocates only the
result and one working vector, plus the derivative of each run's shared values.
"""
_reparam_gradient(backend, reparametrizer, x, g_y) =
    last(value_and_gradient(_reparam_objective, backend, x,
                            Constant(reparametrizer), Constant(g_y)))

# The objective is a TOP-LEVEL function taking its non-differentiated operands as
# `Constant` contexts, not a closure capturing them. Enzyme differentiates the
# callable itself, so a closure over the freshly allocated `g_y` is a callable
# holding mutable data and Enzyme cannot prove it read-only:
#
#     EnzymeMutabilityException: Function argument passed to autodiff cannot be
#     proven readonly.
#
# `AutoEnzyme(; function_annotation = Enzyme.Const)` silences that by marking the
# callable inactive, and that is how callers had to spell the backend. But
# `function_annotation` is a property of the WHOLE backend, so the workaround for
# this one site taxed every gradient. Passing the operands as `Constant` contexts
# instead leaves nothing mutable on the callable, so a bare `AutoEnzyme()` works.
# What the three spellings cost is measured in
# `bench/research/results/annotation_sweep.json` (5 rounds, backend order
# rotated); figures are deliberately not copied here. A single funnel
# measurement once put `Const` at ~1.7x bare; the rotated sweep did not
# replicate it. `Const` stays accepted and is still correct.
#
# Deliberately NOT `prepare_gradient`d. A cached preparation was measured under
# Enzyme and is a wash (1.14x either way, sign flipping with dimension) while
# costing +160 B/gradient just to reach the cache, so the cache buys nothing and
# adds a lifetime to reason about.
_reparam_objective(x_, reparametrizer, g_y) = begin
    ljac_, y_ = reparametrizer(x_)
    ljac_ + dot(g_y, y_)
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
constant number or a callable (a `Function` or a callable struct) applied to the
whole parameter vector, so `x -> x[9]` reads the location off coordinate 9 and
`0.` pins it to zero. A log-scale may combine several coordinates, as in a scale
mixture `beta = s(θ) * z` whose `log(s(θ))` the callable computes.

A callable struct need not subtype `Function` (from WarmupHMC 0.2.15; before it,
one that did not was silently used as a constant). A constant must be `Real`:
[`IndexedReparametrization`](@ref) rejects an argument that is neither a `Real`
nor callable at construction. Mixing constants and callables within ONE rule is
fine; across the pairs of one `IndexedReparametrization`, see its concrete-type
requirement.

The callables are applied to the sampler's SOURCE-coordinate vector as it was on
entry: [`IndexedReparametrization`](@ref) writes its output into a copy, so no
coordinate's rule ever observes another coordinate's transformed value. They also
run under AD on every gradient evaluation (see [`ReparametrizedProblem`](@ref)),
so keep them cheap and type-generic — index, arithmetic, `exp`/`log`, not
`Float64`-annotated code or anything that mutates.

`InverseFunctions.inverse` returns the same rule with `target` and `source`
swapped.

`PartiallyCentered` and the scalar transform come from
ReparametrizableDistributions. Use floating-point centerings such as
`PartiallyCentered(1.0)`: adaptation writes fitted floating-point values back
into the same pairs vector. By default, each restarting warm-up window scores
11 candidates over `range(0, 1, 11)` and updates the source centering; the target
remains fixed. See [`CandidateScoringPlan`](@ref) and [`WindowSelectionPlan`](@ref)
for custom scoring and selection.

See [Reparametrization](@ref) for an executed construction example.
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
    SharedAccessor(f, index)

A location or log-scale accessor that reads element `index` of `f(x)`:
`SharedAccessor(f, i)(x) == f(x)[i]`. Use it when many coordinates take their
location or log-scale from one computation, such as a vector of locations a
model computes in one pass, so that the computation runs once per vector instead
of once per coordinate.

Within an [`IndexedReparametrization`](@ref), consecutive pairs in increasing
index order share an accessor position when their accessors there read the same
`f` (`===`), and WarmupHMC evaluates `f` once for all of them, on every path:
the map and its inverse, the gradient, the transport of retained states to new
centerings, and centering selection. A plain accessor that is the same object in
consecutive pairs, such as one rule reused for every pair of a block, is shared
in the same way without this wrapper. Each accessor position is evaluated once
per such run of pairs, so a location and a log-scale read from one `f` evaluate
it twice; give them separate functions to avoid that.

The shared value is computed where its run of pairs starts, so `f` must not read
the coordinates of the pairs in its own run. The inverse map already requires
this of any accessor (see [`IndexedReparametrization`](@ref)).

Called on its own, a `SharedAccessor` computes `f(x)` in full, so code that
calls accessors directly, such as a [`CandidateScoringPlan`](@ref), keeps
working unchanged.
"""
struct SharedAccessor{F,I}
    f::F
    index::I
end
(a::SharedAccessor)(x) = a.f(x)[a.index]

# --- Runs of pairs that share accessor work ---------------------------------------
#
# Every path that walks the pairs evaluates their accessors once per RUN: a maximal
# sequence of consecutive pairs, in strictly increasing index order, whose accessors
# agree position by position on what they compute (`_share_key`). Accessors applied
# to the whole vector once per pair cost O(N) each, so N pairs sharing one O(N)
# location made every map, gradient, transport and selection O(N^2) (snag
# `use-the-public-i-0660224b`).
#
# The run's shared values ("frames") are evaluated at its first pair. The forward
# map reads the entry vector there, as every pair did before. The inverse reads the
# vector as recovered up to that pair; the inverse's dependency order already
# requires every coordinate an accessor reads to be recovered by then, and a run's
# own coordinates are never read by its shared accessors, so each pair sees the
# value it saw when it evaluated the accessor itself. A run of one pair is exactly
# the old per-pair evaluation. Increasing indices keep each coordinate in a run
# once, so the seed and pool updates of a run can be made together.
_share_key(a) = a
_share_key(a::SharedAccessor) = a.f
_frame(a, x) = maybecall(a, x)
_frame(a::SharedAccessor, x) = a.f(x)
_read_frame(a, frame) = frame
_read_frame(a::SharedAccessor, frame) = frame[a.index]

_keys_match(a::NTuple{N,Any}, b::NTuple{N,Any}) where {N} =
    all(map((p, q) -> _share_key(p) === _share_key(q), a, b))
_keys_match(a, b) = false
_shares_work(a::Reparametrization, b::Reparametrization) = _keys_match(a.args, b.args)
_shares_work(a, b) = false
_index_follows(a::Integer, b::Integer) = a < b
_index_follows(a, b) = false
# Whether `next` continues the run that `previous` belongs to.
_continues_run(previous::Pair, next::Pair) =
    _index_follows(first(previous), first(next)) && _shares_work(last(previous), last(next))
# The last pair of the run starting at `i`, and the first of the run ending at `j`.
function _run_end(pairs, i, stop=lastindex(pairs))
    j = i
    while j < stop && _continues_run(pairs[j], pairs[j + 1])
        j += 1
    end
    j
end
function _run_start(pairs, j, start=firstindex(pairs))
    i = j
    while i > start && _continues_run(pairs[i - 1], pairs[i])
        i -= 1
    end
    i
end

# A rule's shared values at `x`, and the rule evaluated from them. A pair value
# that is not a `Reparametrization` keeps reading the vector itself.
_pair_frames(value::Reparametrization, x) = map(a -> _frame(a, x), value.args)
_pair_frames(value, x) = x
_framed_rargs((;source, args)::Reparametrization, xi, frames) =
    (source, xi, map(_read_frame, args, frames)...)
_framed_rargs((;source, args)::Reparametrization, xi, gi, frames) =
    (source, xi, gi, map(_read_frame, args, frames)...)
_framed_rargs(value, xi, gi, x) = reparam_rargs(value, xi, gi, x)
_reparam_framed(r::Reparametrization, xi, frames) = reparam(r.target, _framed_rargs(r, xi, frames)...)
_reparam_framed(r::Reparametrization, xi, gi, frames) = reparam(r.target, _framed_rargs(r, xi, gi, frames)...)
_reparam_framed(value, xi, x) = reparam(value, xi, x)
_reparam_framed(value, xi, gi, x) = reparam(value, xi, gi, x)

"""
    IndexedReparametrization(pairs)

A vector of `index => Reparametrization(...)` pairs for a
[`ReparametrizedProblem`](@ref). Each index identifies one scalar coordinate in
the unconstrained position vector. Unlisted coordinates pass through unchanged;
the wrapper sums the maps' log-Jacobians. An empty vector applies no map and does
not infer a model's hierarchy.

Warm-up replaces each pair's source centering in place. Independent adaptive
chains need independent problem state; the vector-of-RNGs overload copies a
shared problem, whereas a caller-supplied vector of problems is used as given.
The cooperative sampler shares a reparametrization across its chains and refits
it from pooled evidence.

Build the pairs deterministically in the same order and with the same length
when resuming. Checkpoint controls are restored positionally, so a matching
model dimension does not establish that a reordered map is compatible. During
inversion, a transformed coordinate used by another map's location or log-scale
accessor must precede that dependent map.

The vector must retain a concrete rule type. Mixing constants and differently
typed closures across pairs can widen that type and is rejected at construction.
Closures created at one source location, for example in a comprehension, share a
type. For heterogeneous blocks, one callable struct can represent either a
coordinate lookup or a constant; the [Reparametrization](@ref) guide executes
that construction. Every accessor argument must be a `Real` or callable.

Accessors are evaluated once per run of pairs that share them, not once per
pair: consecutive pairs, in increasing index order, whose location (or
log-scale) is the same accessor object, or a [`SharedAccessor`](@ref) over the
same function, evaluate it once for the whole run on every path. A location
that many coordinates share therefore costs one evaluation per vector. Order the
pairs of such a block consecutively to benefit.
"""
struct IndexedReparametrization{P} <: AbstractReparametrization
    pairs::P
    function IndexedReparametrization(pairs::P) where {P}
        isempty(pairs) || isconcretetype(fieldtype(eltype(pairs), 2)) || throw(ArgumentError(
            "IndexedReparametrization pairs must have a concrete element type; " *
            "avoid mixing constants and differently typed accessor closures in one vector " *
            "(got eltype $(eltype(pairs)); the first pair's rule is a " *
            "$(typeof(last(first(pairs))))). " *
            "Closures written at different places in the source are different types. " *
            "Give every pair one accessor type: closures built at one place (a " *
            "comprehension), or one callable struct that holds either a coordinate " *
            "index or a constant; see the IndexedReparametrization docstring.",
        ))
        isempty(pairs) || _check_accessors(first(pairs))
        new{P}(pairs)
    end
end

# The element type is concrete, so every pair's accessors share the first pair's
# types, and whether a value is callable depends only on its type.
_check_accessors(pair) = _check_accessors(first(pair), last(pair))
_check_accessors(idx, ::Any) = nothing
function _check_accessors(idx, value::Reparametrization)
    for (j, a) in enumerate(value.args)
        _is_accessor(a) || throw(ArgumentError(
            "IndexedReparametrization: argument $(j + 2) of the Reparametrization for " *
            "index $(idx) is a $(typeof(a)), which is neither a Real constant nor " *
            "callable. Each location/log-scale argument is a Real or a callable applied " *
            "to the whole source-coordinate vector, such as `x -> x[9]`.",
        ))
    end
    nothing
end
with_logabsdet_jacobian!(y::AbstractVector, (;pairs)::IndexedReparametrization, x::AbstractVector) = begin
    ljac = 0.
    i = firstindex(pairs)
    while i <= lastindex(pairs)
        j = _run_end(pairs, i)
        frames = _pair_frames(last(pairs[i]), x)
        for k in i:j
            idx, value = pairs[k]
            tmp, y[idx] = _reparam_framed(value, x[idx], frames)
            ljac += tmp
        end
        i = j + 1
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
    pairs = ir.pairs
    i = firstindex(pairs)
    while i <= lastindex(pairs)
        j = _run_end(pairs, i)
        # Read where the run starts: everything its shared accessors read is
        # recovered by then, and none of the run's own coordinates is read.
        frames = _pair_frames(last(pairs[i]), source)
        for k in i:j
            idx, value = pairs[k]
            ljac_i, source_i = _reparam_framed(inverse(value), target[idx], frames)
            source[idx] = source_i
            ljac += ljac_i
        end
        i = j + 1
    end
    ljac, source
end
_inverse_with_logabsdet_jacobian!(source::AbstractVector,
                                  ir::AbstractReparametrization,
                                  target::AbstractVector) =
    with_logabsdet_jacobian!(copyto!(source, target), inverse(ir), target)
_inverse_with_logabsdet_jacobian(ir::AbstractReparametrization,
                                 target::AbstractVector) =
    _inverse_with_logabsdet_jacobian!(copy(target), ir, target)
# `t(x)`, written into `y` instead of a fresh `copy(x)`.
_with_logabsdet_jacobian_into!(y, t, x) = with_logabsdet_jacobian!(copyto!(y, x), t, x)

# --- Controls: what adaptation changes, checkpoints record, transport compares ---
#
# `IndexedReparametrization` keeps its historical representation (a vector of
# `idx => PartiallyCentered` sources). Other reparametrizers define their own
# serializable controls; see `WindowSelectionPlan`.
has_reparametrization(ir::IndexedReparametrization) = !isempty(ir.pairs)
has_reparametrization(::AbstractReparametrization) = true
reparam_controls(ir::IndexedReparametrization) =
    [idx => value.source for (idx, value) in ir.pairs]
restore_reparam_controls!(ir::IndexedReparametrization, controls) =
    (isempty(controls) || (ir.pairs .= _reparam_pairs_with_sources(ir, controls)); ir)
snapshot_reparametrization(ir::IndexedReparametrization) =
    IndexedReparametrization(copy(ir.pairs))
snapshot_reparametrization(ir::AbstractReparametrization) = deepcopy(ir)
_restore_reparametrization!(ir::IndexedReparametrization, old) = (ir.pairs .= old.pairs; ir)
_restore_reparametrization!(ir::AbstractReparametrization, old) =
    restore_reparam_controls!(ir, reparam_controls(old))

_require_indexed(ir) = ir isa IndexedReparametrization || throw(ArgumentError(
    "adapting a $(nameof(typeof(ir))) needs a WindowSelectionPlan; the candidate " *
    "grid and CandidateScoringPlan only apply to an IndexedReparametrization"))
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
# A nonfinite loss marks a candidate the scorer could not evaluate: an observation
# variance that underflowed to zero, a nonfinite scorer output. Bare `argmin`
# prefers `NaN`, so selection committed the first such candidate (snag
# `nonlinear-center-dd5bbc3d`). Only finite losses compete; `nothing` means none
# is finite. Among finite losses the winner is exactly `argmin`'s.
function minimizer((;pairs)::OnlineReparametrizer; kwargs...)
    losses = map(p -> reparametrization_loss(last(p); kwargs...), pairs)
    any(isfinite, losses) || return nothing
    pairs[argmin(i -> isfinite(losses[i]) ? losses[i] : Inf, eachindex(losses))]
end
scale_estimate(or::OnlineReparametrizer; kwargs...) = scale_estimate(last(minimizer(or; kwargs...)))

# The centering coordinate `idx` moves to. With no finite candidate it keeps
# `current`, as `_select_transformation!` keeps the active metric. The selection
# resets the recorder, so nothing else records an excluded candidate: collect
# `idx => kept` in `excluded` for `_warn_excluded_candidates`.
function _minimizing_source!(excluded, or::OnlineReparametrizer, idx, current; kwargs...)
    best = minimizer(or; kwargs...)
    any(p -> !isfinite(reparametrization_loss(last(p); kwargs...)), or.pairs) &&
        push!(excluded, idx => isnothing(best))
    isnothing(best) ? current : first(best)
end

function _warn_excluded_candidates(excluded; limit=5)
    isempty(excluded) && return
    @warn "Centering selection excluded candidates with a nonfinite loss (the scorer could not evaluate them)" n_coordinates=length(excluded) coordinates=first.(first(excluded, limit)) n_kept_centering=count(last, excluded)
end
reset!((;pairs)::OnlineReparametrizer) = (foreach(p -> reset!(last(p)), pairs); nothing)
_mark_group!(loss::WeightedReparametrizationLoss) = (loss.groups += 1; loss)
_mark_group!((;pairs)::OnlineReparametrizer) = foreach(p -> _mark_group!(last(p)), pairs)

function _candidate_scoring_loss_row(pair_number, index, candidate, accumulator, groups)
    (; pair_number,
       index,
       candidate=candidate.c,
       loss=groups > 2 ? reparametrization_loss(accumulator) : missing,
       groups,
       effective_n=iszero(accumulator.weight2) ? missing : effective_n(accumulator))
end

function _candidate_scoring_losses((;pairs)::OnlineReparametrizer)
    [
        _candidate_scoring_loss_row(
            pair_number, index, candidate, accumulator,
            OnlineStatsBase.nobs(candidates),
        )
        for (pair_number, (index, candidates)) in enumerate(pairs)
        for (candidate, accumulator) in candidates.pairs
    ]
end

reparametrization_candidates(::PartiallyCentered; n=11) = Iterators.map(PartiallyCentered, range(0, 1, n))
OnlineReparametrizer((;source)::Reparametrization, xg...; kwargs...) = OnlineReparametrizer(source, xg...; kwargs...)
OnlineReparametrizer(source::PartiallyCentered, xg...;
                     accumulator=OnlineReparametrizationLoss, kwargs...) = OnlineReparametrizer([
    target => accumulator(xg...)
    for target in reparametrization_candidates(source; kwargs...)
])

# --- Batch optimization ---

optimize!(ir::IndexedReparametrization, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    _select_on_pool!(ir.pairs, k -> OnlineReparametrizer(last(ir.pairs[k]), xg...; kwargs...),
                     xg, loss_kwargs)
    IndexedReparametrization(ir.pairs)
end

# The pool search: score every candidate centering of each pair on the pool
# columns, choose the pair's new centering, and move that coordinate of every
# column into it, pair after pair in order, so a later pair's accessors read the
# coordinates earlier pairs have moved. A run of pairs sharing accessor work
# (`_run_end`) is scored and moved together, reading its shared values once per
# column: its shared accessors read none of its own coordinates, and each pair
# reads and moves only its own, so a pair's observations and its move are
# exactly what the pair-by-pair order gives. `accumulator_for(k)` is pair `k`'s
# candidate accumulator.
function _select_on_pool!(pairs, accumulator_for, xg, loss_kwargs)
    excluded = Pair{Any,Bool}[]
    columns = zip(eachcol.(xg)...)
    i = firstindex(pairs)
    while i <= lastindex(pairs)
        j = _run_end(pairs, i)
        run = i:j
        old = [last(pairs[k]) for k in run]
        accumulators = [accumulator_for(k) for k in run]
        for xgi in columns
            frames = _pair_frames(first(old), first(xgi))
            for (n, k) in enumerate(run)
                OnlineStatsBase.fit!(accumulators[n], _framed_rargs(
                    old[n], getindex.(xgi, first(pairs[k]))..., frames)...)
            end
        end
        moved = falses(length(run))
        for (n, k) in enumerate(run)
            OnlineStatsBase.nobs(accumulators[n]) > 2 || continue
            idx, value = first(pairs[k]), old[n]
            new_source = _minimizing_source!(excluded, accumulators[n], idx, value.source; loss_kwargs...)
            pairs[k] = idx => Reparametrization(value.target, new_source, value.args...)
            moved[n] = true
        end
        if any(moved)
            for xgi in columns
                frames = _pair_frames(first(old), first(xgi))
                for (n, k) in enumerate(run)
                    moved[n] || continue
                    idx, value = pairs[k]
                    trans = Reparametrization(value.source, old[n].source, value.args...)
                    setindex!.(xgi, _reparam_framed(trans, getindex.(xgi, idx)..., frames)[2:end], idx)
                end
            end
        end
        i = j + 1
    end
    _warn_excluded_candidates(excluded)
    pairs
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
    pairs = ir.pairs
    stop = min(lastindex(pairs), lastindex(ors.pairs))
    i = firstindex(pairs)
    while i <= stop
        j = _run_end(pairs, i, stop)
        frames = _pair_frames(last(pairs[i]), position)
        for k in i:j
            idx, value = pairs[k]
            stored_idx, or = ors.pairs[k]
            idx == stored_idx || throw(ArgumentError(
                "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
            ))
            OnlineStatsBase.fit!(
                or,
                _framed_rargs(value, position[idx], gradient[idx], frames)...;
                weight, count,
            )
        end
        i = j + 1
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
    pairs = ir.pairs
    stop = min(lastindex(pairs), lastindex(ors.pairs))
    i = firstindex(pairs)
    while i <= stop
        j = _run_end(pairs, i, stop)
        # The direct scorer's shared values, computed only if the plan declines a
        # candidate of this run, and then once for the whole run.
        shared = nothing
        for k in i:j
            pair_number = k - firstindex(pairs) + 1
            idx, value = pairs[k]
            stored_idx, or = ors.pairs[k]
            idx == stored_idx || throw(ArgumentError(
                "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
            ))
            for (candidate, accumulator) in or.pairs
                observation = plan.score(frame, pair_number, idx, value, candidate)
                if isnothing(observation)
                    isnothing(shared) && (shared = _pair_frames(value, position))
                    observation = reparam(
                        candidate,
                        _framed_rargs(value, position[idx], gradient[idx], shared)...,
                    )
                end
                OnlineStatsBase.fit!(accumulator, observation; weight, count)
            end
        end
        i = j + 1
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
    excluded = Pair{Any,Bool}[]
    ir.pairs .= Base.broadcasted(ir.pairs, ors.pairs) do (idx, value), (stored_idx, or)
        idx == stored_idx || throw(ArgumentError(
            "nonlinear accumulator index $stored_idx does not match reparametrizer index $idx",
        ))
        OnlineStatsBase.nobs(or) > 2 || return idx => value
        new_source = _minimizing_source!(excluded, or, idx, value.source; loss_kwargs...)
        idx => Reparametrization(value.target, new_source, value.args...)
    end
    _warn_excluded_candidates(excluded)
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
                           good_leaf_threshold=log(1e-2), capacity=1000)
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
        _nonlinear_accumulator(candidate_scoring_plan(lpdf), reparametrizer(lpdf),
                               LogDensityProblems.dimension(lpdf), capacity),
    )
end

"""
    WeightedLeafSample(dimension, capacity)

`capacity` states drawn WITH replacement from a weighted stream: after any
prefix of the stream, every slot independently holds state `i` with
probability `w_i / sum(w)`. A new state replaces each slot with probability
`w / sum(w)`; geometric skips visit only the replaced slots.
"""
mutable struct WeightedLeafSample
    position::Matrix{Float64}
    gradient::Matrix{Float64}
    total::Float64
    rng::Random.Xoshiro
end
WeightedLeafSample(dimension, capacity) = WeightedLeafSample(
    zeros(dimension, capacity), zeros(dimension, capacity), 0.0, Random.Xoshiro(0x5eed))

function _observe_sample!(sample::WeightedLeafSample, position, gradient, weight)
    sample.total += weight
    p = weight / sample.total
    slots = size(sample.position, 2)
    slot = 0
    while true
        if p < 1
            skip = log(1 - rand(sample.rng)) / log1p(-p)   # misses before the next hit
            skip < slots - slot || break
            slot += floor(Int, skip) + 1
        else
            slot += 1
            slot <= slots || break
        end
        sample.position[:, slot] .= position
        sample.gradient[:, slot] .= gradient
    end
    sample
end
reset!(sample::WeightedLeafSample) = (sample.total = 0.0; sample)
_sample_evidence(sample::WeightedLeafSample) = sample.total > 0 ?
    (sample.position, sample.gradient) :
    (sample.position[:, 1:0], sample.gradient[:, 1:0])

_nonlinear_accumulator(::WindowSelectionPlan, ir, dimension, capacity) =
    WeightedLeafSample(dimension, capacity)
# A fixed (`nonlinear_adapt=false`) custom reparametrizer records nothing;
# adapting one without a selection plan is refused by
# `_check_nonlinear_adaptable` before sampling starts.
_nonlinear_accumulator(plan, ir, dimension, capacity) = ir isa IndexedReparametrization ?
    OnlineReparametrizer(ir; accumulator=WeightedReparametrizationLoss) : nothing

# Fail before sampling rather than at the first restarting window, which a run
# may never reach.
_check_nonlinear_adaptable(lpdf, nonlinear_adapt) = begin
    ir = reparametrizer(lpdf)
    nonlinear_adapt && has_reparametrization(ir) &&
        !(candidate_scoring_plan(lpdf) isa WindowSelectionPlan) && _require_indexed(ir)
    nothing
end

_observe_nonlinear!(plan, ir, online, position, gradient, weight) =
    OnlineStatsBase.fit!(ir, online, position, gradient;
                         weight, count=false, scoring_plan=plan)
_observe_nonlinear!(::WindowSelectionPlan, ir, online, position, gradient, weight) =
    (_observe_sample!(online, position, gradient, weight); nothing)
_mark_nonlinear_group!(plan, online) = _mark_group!(online)
_mark_nonlinear_group!(::WindowSelectionPlan, online) = nothing

"""
    candidate_scoring_losses(payload::NamedTuple)
    candidate_scoring_losses(lpdf, positions, gradients; weights=nothing)

Return one row per reparametrized coordinate and candidate, with fields
`pair_number`, `index`, `candidate`, `loss`, `groups`, and `effective_n`.
`candidate` is the candidate partial-centering value on the fixed
`0.0:0.1:1.0` grid. `loss` is `missing` until more than two evidence groups
have accumulated, matching the sampler's selection threshold.

For finite, nondegenerate observations, the default candidate objective is

```
L = w₁ * (-mean(Δlog|J|) + log(var(position)) / 2) +
    (1 - w₁) * cor(position, gradient),       w₁ = 0
```

so the selected candidate minimizes weighted position-gradient correlation.
Writing `W = sum(wᵢ)`, `q̄ = sum(wᵢqᵢ) / W`, and
`ḡ = sum(wᵢgᵢ) / W`, that correlation is exactly

```
sum(wᵢ * (qᵢ - q̄) * (gᵢ - ḡ)) /
sqrt(sum(wᵢ * (qᵢ - q̄)^2) * sum(wᵢ * (gᵢ - ḡ)^2))
```

The log-Jacobian/log-variance term is present but has zero default weight; a
`CandidateScoringPlan` changes how candidate observations are constructed, not
this criterion. WarmupHMC does not reject nonfinite candidate observations or
losses. The implementation evaluates both displayed terms before applying the
weights, so degenerate or nonfinite input can produce a nonfinite loss even at
`w₁ = 0`; this function returns that value unchanged. The sampler's selection
never picks such a candidate: it minimizes over the finite losses only, keeps a
coordinate's current centering when none of its losses is finite, and warns
which coordinates had a candidate excluded.

The `payload` method reads the serialized `nonlinear_recorder` accumulator from
an adaptive or cooperative checkpoint. It is the actual pending online curve
since the most recent nonlinear reset, not a history: after a restarting window
selects a candidate, the recorder is reset before that boundary's checkpoint is
written. Payloads without that recorder, including cooperative checkpoints (the
merged sampler refits from pooled halos and keeps no recorder), raise an
`ArgumentError`.

The matrix method deliberately recomputes a retrospective curve. `positions`
and `gradients` must have identical `dimension × observations` shapes and be in
the current sampler/source coordinate frame. The attached
`CandidateScoringPlan`, when present, prepares its own candidate frame from
each column. Each positive-weight column counts as one evidence group; zero
weights are ignored and negative weights raise `ArgumentError`. This is the
same scorer and criterion, but it is the recorded warm-up curve only if the
caller supplies the exact online leaf stream, weights, and group boundaries.
Ordinary saved halo or posterior matrices do not contain those boundaries, so
their curves are retrospective diagnostics.
"""
candidate_scoring_losses(recorder::NonlinearRecorder) =
    _candidate_scoring_losses(recorder.online)

function candidate_scoring_losses(payload::NamedTuple)
    hasproperty(payload, :nonlinear_recorder) || throw(ArgumentError(
        "checkpoint payload has no nonlinear_recorder candidate-scoring state",
    ))
    recorder = payload.nonlinear_recorder
    recorder isa NonlinearRecorder || throw(ArgumentError(
        "checkpoint nonlinear_recorder has unsupported type $(typeof(recorder))",
    ))
    candidate_scoring_losses(recorder)
end

function candidate_scoring_losses(lpdf,
                                  positions::AbstractMatrix,
                                  gradients::AbstractMatrix;
                                  weights::Union{Nothing,AbstractVector}=nothing)
    size(positions) == size(gradients) || throw(DimensionMismatch(
        "positions and gradients must have identical shapes; got " *
        "$(size(positions)) and $(size(gradients))",
    ))
    dimension = LogDensityProblems.dimension(lpdf)
    size(positions, 1) == dimension || throw(DimensionMismatch(
        "positions and gradients must have $dimension rows; got $(size(positions, 1))",
    ))
    n_observations = size(positions, 2)
    if !isnothing(weights)
        length(weights) == n_observations || throw(DimensionMismatch(
            "weights must have $n_observations entries; got $(length(weights))",
        ))
    end

    ir = reparametrizer(lpdf)
    online = OnlineReparametrizer(
        ir; accumulator=WeightedReparametrizationLoss,
    )
    plan = candidate_scoring_plan(lpdf)
    if isnothing(weights)
        for (position, gradient) in zip(eachcol(positions), eachcol(gradients))
            OnlineStatsBase.fit!(
                ir, online, position, gradient; scoring_plan=plan,
            )
        end
    else
        for (position, gradient, weight) in
            zip(eachcol(positions), eachcol(gradients), weights)
            OnlineStatsBase.fit!(
                ir, online, position, gradient; weight, scoring_plan=plan,
            )
        end
    end
    _candidate_scoring_losses(online)
end

_uses_online_candidate_scoring(::DirectCandidateScoring) = false
_uses_online_candidate_scoring(::CandidateScoringPlan) = true
_uses_online_candidate_scoring(::WindowSelectionPlan) = false

# `recorder.mode` is the DECLARED evidence mode; this returns the EFFECTIVE one.
# A custom scoring plan reads `:linear_pool` as online all-good-leaf evidence and
# never replays the retained linear pool, so the two disagree in exactly one
# case: a plan is attached AND the declared mode is `:linear_pool`.
#
# The remap is silent, which is what makes it worth a note here rather than at
# each caller. It matters most for CHECKPOINTS. A payload records the declared
# mode (`nonlinear_recorder.mode`) and the plan marker
# (`custom_candidate_scoring`) as two separate keys and never the effective
# mode, so a consumer that reads `mode` alone can be off by this remap.
#
# The two keys ARE sufficient to recover it: `_uses_online_candidate_scoring`
# and `_has_custom_candidate_scoring` are the same predicate spelled twice, so a
# consumer re-applies the rule below and needs nothing else.
#
# An `effective_nonlinear_evidence` key is deliberately NOT stored. It would be
# a derived value sitting beside both of its own inputs, which nothing forces to
# agree — and it fails in the silent direction: change this rule and every older
# checkpoint carries a value computed by the previous one, indistinguishable
# from a current one. The inputs are the data; this is a function of them.
#
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
    has_reparametrization(ir) || return recorder
    plan = candidate_scoring_plan(lpdf)
    plan isa WindowSelectionPlan || _require_indexed(ir)
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
        _observe_nonlinear!(
            plan,
            ir,
            recorder.online,
            @view(leaves.position[:, i]),
            @view(leaves.gradient[:, i]),
            weight,
        )
        recorded = true
    end
    recorded && _mark_nonlinear_group!(plan, recorder.online)
    recorder
end

reset!(recorder::NonlinearRecorder) = (reset!(recorder.online); recorder)
OnlineStatsBase.fit!(ir::IndexedReparametrization, ors::OnlineReparametrizer, xg::AbstractMatrix...; loss_kwargs=(;), kwargs...) = begin
    length(ors.pairs) == length(ir.pairs) || throw(DimensionMismatch(
        "$(length(ors.pairs)) candidate accumulators for $(length(ir.pairs)) reparametrization pairs"))
    _select_on_pool!(ir.pairs, k -> last(ors.pairs[k]), xg, loss_kwargs)
    ir
end

# --- Hooks for adaptive_warmup_mcmc ---

_reparametrization_ad_backend(p::ReparametrizedProblem) = p.ad_backend
_reparametrization_ad_backend(p::WrappedLogDensityProblem) =
    _reparametrization_ad_backend(parent(p))

# Same shape as `_reparam_objective` above, and for the same reason: a top-level
# function with `Constant` operands rather than a callable carrying them. A struct
# is no safer than a closure here — it is immutable, but its `old_gradient` field
# is an Array, so Enzyme still cannot prove the callable read-only. Both AD sites
# had to move together: `function_annotation` is set on the BACKEND, so leaving
# this one unfixed would keep the annotation mandatory and go on taxing every
# gradient through the hot path.
#
# The two maps write into `Cache` work vectors rather than fresh copies, so one
# gradient preparation serves every column of a transport without allocating:
# each copy was a d-vector, and its Enzyme shadow another, per halo column
# (snag `find-reparametri-e2a5b2a7`).
_transport_objective!(new_position, model_position, old_position,
                      new_ir, old_ir, old_gradient) = begin
    ljac_new, _ = _with_logabsdet_jacobian_into!(model_position, new_ir, new_position)
    ljac_old, _ = _inverse_with_logabsdet_jacobian!(old_position, old_ir, model_position)
    ljac_new + ljac_old + dot(old_gradient, old_position)
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

Every column is computed in a fixed set of d-vectors, allocated and prepared
once per call, so the cost in memory is independent of the number of columns.
Each column is read completely before it is written, so `position` may be
`old_position`.
"""
function _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                                  position, gradient)
    size(position, 2) == 0 && return gradient
    new_ir = reparametrizer(lpdf)
    backend = _reparametrization_ad_backend(lpdf)
    # Differentiate at a plain Vector, not the halo column view: Enzyme on Julia
    # 1.13 cannot prove the storage activity of a view into the ElasticMatrix
    # halo, nor of the ElasticVector a copy of one becomes
    # (EnzymeRuntimeActivityError).
    model_position, x, g_old, g_new =
        ntuple(_ -> Vector{eltype(position)}(undef, size(position, 1)), 4)
    # Prepared at the first column's values, for backends whose preparation
    # evaluates the objective.
    _transport_position!(x, model_position, new_ir, old_ir, view(old_position, :, 1))
    copyto!(g_old, view(old_gradient, :, 1))
    transport_gradient! = _transport_gradient_method(backend, new_ir, old_ir, x, g_old)
    columns = zip(eachcol(old_position), eachcol(old_gradient),
                  eachcol(position), eachcol(gradient))
    for (x_old_j, g_old_j, x_new_j, g_new_j) in columns
        _transport_position!(x_new_j, model_position, new_ir, old_ir, x_old_j)
        copyto!(x, x_new_j)
        copyto!(g_old, g_old_j)
        transport_gradient!(g_new, x, g_old)
        g_new_j .= g_new
    end
    gradient
end

"""
    WarmupHMC._transport_gradient_method(backend, new_ir, old_ir, x, g_old)

Return `f!(g_new, x, g_old)`, which writes into `g_new` the gradient in `x` of
`_transport_objective!` for the vectors `x` and `g_old` it is called with. Those
are the same two vectors on every call; their contents change per column.

This method prepares one DifferentiationInterface gradient of the whole objective.
With a reverse-mode `AutoEnzyme` backend and two
[`IndexedReparametrization`](@ref)s, the Enzyme extension differentiates it pair
by pair instead, as for [`_reparam_gradient`](@ref).
"""
function _transport_gradient_method(backend, new_ir, old_ir, x, g_old)
    contexts = (Cache(similar(x)), Cache(similar(x)),
                Constant(new_ir), Constant(old_ir), Constant(g_old))
    prep = prepare_gradient(_transport_objective!, backend, x, contexts...)
    (g_new, x, _) -> value_and_gradient!(_transport_objective!, g_new, prep, backend, x, contexts...)
end
_transport_position!(x_new, model_position, new_ir, old_ir, x_old) = begin
    _with_logabsdet_jacobian_into!(model_position, old_ir, x_old)
    _inverse_with_logabsdet_jacobian!(x_new, new_ir, model_position)
end

# A centering switch changes coordinate meaning. Preserve the active physical
# point, just as `_jointly_transport_halo!` preserves the stored history, then
# refresh the density and gradient in the new source frame. Re-evaluating the
# old source vector without transforming it can send the chain far outside the
# typical set at every restarting window.
function _transport_active_evaluation(lpdf, old_ir, position_and_gradient)
    _, model_position = old_ir(position_and_gradient.q)
    _, source_position = _inverse_with_logabsdet_jacobian(reparametrizer(lpdf), model_position)
    DynamicHMC.evaluate_ℓ(lpdf, source_position; strict=false)
end

# A candidate can be representable in the streaming scoring evidence but not
# in every point of the linear-adaptation pool (e.g. HSGP scales underflow).
# Treat the coordinate update as a transaction: an unusable transport must not
# poison the metric, nor change the meaning of the preserved sampler state.
function _validated_transport!(lpdf, old_ir, old_position, old_gradient,
                               position, gradient, position_and_gradient)
    _jointly_transport_halo!(lpdf, old_ir, old_position, old_gradient,
                             position, gradient)
    ir = reparametrizer(lpdf)
    if all(isfinite, position) && all(isfinite, gradient)
        _, physical = old_ir(position_and_gradient.q)
        _, source = _inverse_with_logabsdet_jacobian(ir, physical)
        if all(isfinite, source)
            point = DynamicHMC.evaluate_ℓ(lpdf, source; strict=false)
            if isfinite(point.ℓq) && all(isfinite, point.∇ℓq)
                return point
            end
        end
    end
    bad_positions = count(!isfinite, position)
    bad_gradients = count(!isfinite, gradient)
    _restore_reparametrization!(ir, old_ir)
    _synchronize_scoring!(lpdf)
    position .= old_position
    gradient .= old_gradient
    @warn "Rejected centering update: nonfinite coordinate transport; retaining previous coordinates" bad_positions bad_gradients
    position_and_gradient
end

find_reparametrization!(lpdf, halo_position, halo_gradient, position_and_gradient) = begin
    ir = reparametrizer(lpdf)
    has_reparametrization(ir) || return position_and_gradient
    _require_indexed(ir)
    old_ir = IndexedReparametrization(copy(ir.pairs))
    old_position = copy(halo_position)
    old_gradient = copy(halo_gradient)
    optimize!(ir, halo_position, halo_gradient)
    _synchronize_scoring!(lpdf)
    _validated_transport!(lpdf, old_ir, old_position, old_gradient,
                          halo_position, halo_gradient, position_and_gradient)
end

function find_reparametrization!(lpdf, recorder::NonlinearRecorder,
                                  halo_position, halo_gradient,
                                  position_and_gradient)
    plan = candidate_scoring_plan(lpdf)
    plan isa WindowSelectionPlan && return _select_reparametrization!(
        plan, lpdf, recorder, halo_position, halo_gradient, position_and_gradient,
    )
    _effective_nonlinear_evidence(recorder, lpdf) === :linear_pool &&
        return find_reparametrization!(
        lpdf, halo_position, halo_gradient, position_and_gradient,
    )
    ir = reparametrizer(lpdf)
    has_reparametrization(ir) || return position_and_gradient
    _require_indexed(ir)
    old_ir = IndexedReparametrization(copy(ir.pairs))
    old_position = copy(halo_position)
    old_gradient = copy(halo_gradient)
    optimize!(ir, recorder.online)
    _synchronize_scoring!(lpdf)
    point = _validated_transport!(lpdf, old_ir, old_position, old_gradient,
                                  halo_position, halo_gradient, position_and_gradient)
    reset!(recorder)
    point
end

# A selection plan's boundary step. `:linear_pool` hands over the retained pool;
# the streaming modes filled `recorder.online` during the window. The same
# transaction as the candidate path follows: snapshot, select, synchronize,
# transport (or roll back on a nonfinite transport), then clear the sample.
function _select_reparametrization!(plan::WindowSelectionPlan, lpdf,
                                    recorder::NonlinearRecorder,
                                    halo_position, halo_gradient,
                                    position_and_gradient)
    ir = reparametrizer(lpdf)
    has_reparametrization(ir) || return position_and_gradient
    old_ir = snapshot_reparametrization(ir)
    old_position = copy(halo_position)
    old_gradient = copy(halo_gradient)
    evidence = _effective_nonlinear_evidence(recorder, lpdf) === :linear_pool ?
        (old_position, old_gradient) : _sample_evidence(recorder.online)
    changed = plan.select!(ir, evidence...) === true
    _synchronize_scoring!(lpdf)
    point = changed ? _validated_transport!(
        lpdf, old_ir, old_position, old_gradient,
        halo_position, halo_gradient, position_and_gradient,
    ) : position_and_gradient
    reset!(recorder.online)
    point
end

# Draws are stored in the SAMPLING parametrization: `logdensity` receives the
# sampler position in `source` coordinates and maps it through `ir`
# (source -> target) before handing `y` to the inner problem. Reporting draws in
# the model's own (`target`) parametrization therefore applies `ir` itself.
# Applying `inverse(ir)` here maps target -> source, i.e. the wrong direction —
# it was a no-op only while adaptation left `source == target`.
reparametrize!(lpdf, posterior_position) = begin
    ir = reparametrizer(lpdf)
    has_reparametrization(ir) || return
    for col in eachcol(posterior_position)
        ljac, y = ir(col)
        col .= y
    end
end
