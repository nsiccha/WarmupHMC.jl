# Nonlinear reparametrization

Hierarchical models have a well-known pathology: written in their natural
("centered") form, a group-level coordinate's scale is itself a parameter, so the
geometry the sampler sees changes shape as that scale moves. The textbook fix is
to rewrite the model in a "non-centered" form. Which of the two is better is
model- *and* data-dependent, and for a model with many groups the answer need not
be the same for every group.

WarmupHMC can adapt that choice during warm-up, per coordinate and continuously
rather than as an either/or, by giving each reparametrized coordinate a centering
[`PartiallyCentered`](@ref)`(c)` with `c ∈ [0, 1]` — `1` centered, `0`
non-centered — and re-fitting `c` at warm-up window boundaries.

!!! warning "This page makes no claim about sampling efficiency"
    Nothing here says the reparametrization is faster, cheaper or better-mixing
    than not using it. Those numbers are not in yet. What is documented below is
    what the machinery *does*.

## It does nothing until you build it

This is the first thing to know, and it is easy to get wrong in a way that looks
like success:

```julia
WarmupHMC.reparametrizer(::Any) = IndexedReparametrization([])
```

Every problem that is not a [`ReparametrizedProblem`](@ref) reports an **empty**
reparametrization, an empty one is a **no-op**, and `nonlinear_adapt=true` — the
default on every sampler — then adapts nothing at all. There is no automatic
detection of hierarchical structure. A run with `nonlinear_adapt=true` on a plain
log-density problem is byte-for-byte a run without it.

To get any reparametrization you must supply, yourself:

* the **raw integer indices** into the unconstrained parameter vector of the
  coordinates to reparametrize — one entry per scalar coordinate, not per model
  parameter;
* for each, a **location** and a **log-scale**, either as constants or as
  closures over the whole parameter vector;
* an **AD backend**, used for the transform's own Jacobian. This page uses
  Enzyme, spelled as in the warning below.

On that last point: WarmupHMC differentiates a scalar objective in the *full*
parameter vector (see [`ReparametrizedProblem`](@ref)), which is the shape
reverse mode exists for — its cost is one pass regardless of dimension, while
forward mode costs one pass per input. That is an argument about operation
counts, and it is worth knowing that **the measured wall-clock does not follow
it**: reverse mode wins on small targets here and loses on larger ones, in the
opposite direction to what the counting argument suggests. The table is in the
[`ReparametrizedProblem`](@ref) docstring. Reverse mode is a reasonable default
and what the examples below use; if the gradient is your bottleneck, measure
both on your own model rather than reasoning from dimension.

**The interface ships; the backend is yours to bring.**
`DifferentiationInterface` is a hard dependency of WarmupHMC, so
`ReparametrizedProblem` can always *talk* to a backend — but a backend object
only works once you have loaded the package behind it, and `AutoEnzyme` needs
`using Enzyme`. Nothing in `src/` ever constructs one: `ad_backend` is a field
you fill in.

!!! warning "A bare `AutoEnzyme()` does not work — it needs two options, for two different failures"
    Pass both:

    ```julia
    AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                 function_annotation=Enzyme.Const)
    ```

    They guard different call sites and they fail at different moments, so it is
    worth knowing which is which.

    **`function_annotation=Enzyme.Const`** — without it, the run dies on the
    **first gradient**:

    ```
    EnzymeMutabilityException: Function argument passed to autodiff cannot be
    proven readonly.
    ```

    What gets differentiated is a closure capturing the frozen inner gradient
    `g_y` and the reparametrizer (see [`ReparametrizedProblem`](@ref)), and
    Enzyme will not assume on its own that captured state holds no derivative
    data. `Const` states what is already true here: `g_y` is frozen by
    construction — a constant of the differentiation, not a function of `x`.
    **Do not take Enzyme's own suggestion of `Enzyme.Duplicated`**; it computes
    the same answer, but it allocates a shadow copy of the closure on every
    call. That hint diagnoses the problem; it is not the fix. How much the
    shadow copy costs depends strongly on the target — from ~11× per gradient on
    a 10-dimensional funnel down to nothing measurable on the radon models;
    [`ReparametrizedProblem`](@ref) carries the measured table. `Const` is the
    right annotation regardless, because it is the *correct* one.

    **`set_runtime_activity`** — without it, the run gets past construction and
    past the first gradient, and then dies at the **first restarting window**:

    ```
    EnzymeRuntimeActivityError: Detected potential need for runtime activity.
    ```

    That is the final joint halo transport described under
    [What warm-up actually does](@ref), which runs your backend over a *second*
    closure. Enzyme cannot statically prove that one's activity either, and
    unlike the first it is not fixable by an annotation at the call site.

!!! note "WarmupHMC does not depend on ForwardDiff, and does not want to"
    `Project.toml` has no ForwardDiff entry — not a direct dependency, and there
    is no `[weakdeps]` section for it to hide in either. ForwardDiff reaches the
    environment only transitively, through Pathfinder, and warm-up *actively
    defuses* it — the Pathfinder initialization passes `adtype=NoAD()` so
    Pathfinder's forward-mode default cannot call your `logdensity` with
    `Dual`-valued parameters, which a natively-backed target such as a BridgeStan
    model could not accept. If you find `AutoForwardDiff()` named in a test or a
    benchmark here, that is a harness pinned to its own frozen baselines, not a
    recommendation — `web/src/test/ad_backend.jl` says so at the point of use.

## A complete worked example

Neal's funnel — `v ~ Normal(0, 3)`, `xᵢ ~ Normal(0, exp(v/2))` — is the smallest
model where this matters. Coordinate `1` is `v`; coordinates `2:6` are the `xᵢ`,
each with location `0` and log-scale `v/2`.

```julia
using WarmupHMC, LogDensityProblems, Random
using Enzyme                                       # backs AutoEnzyme
using DifferentiationInterface: AutoEnzyme

struct Funnel
    k::Int
end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]
    -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
end
LogDensityProblems.logdensity_and_gradient(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]
    g = similar(x)
    g[1] = -v / 9 + 0.5 * exp(-v) * sum(abs2, xs) - f.k / 2
    g[2:end] .= .-xs .* exp(-v)
    (LogDensityProblems.logdensity(f, x), g)
end

k = 5
funnel = Funnel(k)

# One `Reparametrization` per reparametrized COORDINATE. `target` is the
# parametrization the model is written in and never moves; `source` is what the
# sampler works in and is what warm-up re-fits. Starting them equal means
# "start where the model is written, and adapt from there".
ir = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                           0., x -> x[1] / 2)
    for i in 2:(k + 1)
])

rp = ReparametrizedProblem(ir, funnel,
    AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                 function_annotation=Enzyme.Const))
result = adaptive_warmup_mcmc(Xoshiro(20260728), rp; n_draws=1000, progress=nothing)
```

The fitted centerings live on the `IndexedReparametrization` you passed in — the
single-chain method mutates it in place — so read them back off `ir`:

```julia
julia> [v.source.c for (_, v) in ir.pairs]
5-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
```

All five coordinates were driven from `1.0` to `0.0`: warm-up found the
non-centered parametrization of the funnel on its own, without being told that
the funnel is the model it was looking at.

Reading the result back off your own object is a *single-chain* affordance. The
multi-chain method gives each chain its own `deepcopy`, so the object you built
stays exactly as you built it and each chain's fitted centerings live on its own
copy — see "What warm-up actually does" below.

`result.posterior_position` is `6 × 1000` and is in the **model's own**
parametrization — warm-up applies the fitted transform to the draws before
returning them, so nothing downstream has to know a reparametrization happened.

!!! note "Provenance of the figures above"
    They were produced by the example exactly as written, at `ba6b4f0`. The same
    run was also executed under `AutoEnzyme(; function_annotation=Enzyme.Duplicated)`
    and under `AutoForwardDiff()` — both as controls, not as recommendations. All
    three return the identical gradient at the first step and the identical
    `[0.0, 0.0, 0.0, 0.0, 0.0]` and `6 × 1000` at the end. What a gradient
    *costs* is what the backend choice decides; what it *returns* here is
    backend-independent.

    Adaptation behaviour is *not* fixed across commits, however: these same figures
    were materially different a few commits earlier. If you are on a different tip,
    re-run rather than assume.

The reparametrization is re-fitted only at warm-up windows that **restart**, and
a window restarts only while the marginal-scale condition number is at or above
`variance_cond_target` (default `2.0`). In the run above, windows 1 and 2 restart
(condition number `2.08`, then `3.44`) and windows 3 and 4 do not (`1.0`), so all
five centerings are already at `0.0` by the end of the *first* window and the
remaining windows sample at the parametrization that was found.

### How long a run does adaptation need?

**On this model, not a long one.** Re-running the example unchanged at
`n_draws=200`, and again at `n_draws=100`, gives the same two restarting windows
and the same final `[0.0, 0.0, 0.0, 0.0, 0.0]`. So the centerings here are settled
well inside a run short enough to use as a smoke test — you do not have to budget
a long run just to find out whether your spec does anything.

Do not read that as a guarantee about the method. It is a property of this
funnel: on a model whose condition number starts below `variance_cond_target`, no
window restarts and the centerings never move at all, however long you sample.
What generalizes is the *mechanism*, not the window count — so if you are
checking that your spec is wired up correctly, watch the boundaries rather than
assuming a re-fit happened:

```julia
adaptive_warmup_mcmc(rng, rp; n_draws=1000, progress=nothing,
    callback=(state, stage) -> begin
        @info "boundary" stage state.outer_counter state.restart state.variance_cond
        false
    end)
```

## Building a spec for a real model

For anything bigger than the funnel, the work is not the `Reparametrization`
objects — it is mapping model structure onto raw integer positions in the
unconstrained vector. `web/src/posteriordb_reparametrizations.jl` in this
repository does exactly that for a set of PosteriorDB posteriors, and is the
most honest picture of the current UX. Eight schools, where the offsets are
constants:

```julia
c = endswith(posterior_name, "noncentered") ? 0. : 1.
1:8 .=> Ref(Reparametrization(
    PartiallyCentered(c),
    PartiallyCentered(c),
    x->x[9],
    x->x[10]
))
```

and a partially-pooled radon model, where they are computed from the Stan data:

```julia
J = stan_jdata["J"]
c = endswith(posterior_name, "noncentered") ? 0. : 1.
(l, s, o) = (J+1, J+2, 0)
map(1:J) do i
    idx = o + i
    idx=>Reparametrization(
        PartiallyCentered(c),
        PartiallyCentered(c),
        x->x[l],
        x->x[s]
    )
end
```

Note what that file has to encode by hand for every posterior: the block of
coordinates, the offset it starts at, and the two positions holding its location
and log-scale. Get one offset wrong and you will reparametrize the wrong
coordinates against the wrong scale — quietly, because nothing validates that a
"log-scale" index really holds a log-scale. The whole file is worth reading
before writing a spec of your own; every branch returns something that
`IndexedReparametrization` accepts directly, and the `else` branch returns
`Nothing[]`, i.e. the no-op.

`target` and `source` are set to the same `c` in every branch there, and `c` is
picked to match how the Stan model is written (`0.` for the `_noncentered`
variants, `1.` otherwise). That is the general rule: **`target` must describe the
parametrization the wrapped log density actually expects.** It is never adapted,
and getting it wrong does not error — it silently samples a different model.

## What warm-up actually does

At the end of a warm-up window that restarts, and immediately before the linear
metric is re-selected:

1. Each reparametrized coordinate is scored against a fixed grid of **11**
   candidate centerings, `range(0, 1, 11)`.
2. Scoring runs over the recorded **halo** — the intermediate NUTS states kept
   during the window (one per trajectory, drawn from the exact marginal proposal
   probabilities over that trajectory's leaves, up to `recording_target` of
   them), not the accepted draws.
3. Each candidate accumulates, online, the covariance of (its transformed
   position, its transformed gradient); the candidate minimizing their
   correlation wins. A coordinate whose marginal is standard-normal has position
   and gradient exactly anti-correlated, so the minimum is the most
   standard-normal-looking candidate.
4. The winner replaces that coordinate's `source`, and the halo states for that
   coordinate are transported into the new parametrization before the next
   coordinate is scored.
5. A coordinate with 2 or fewer halo states keeps the centering it had.
6. Once every coordinate has settled, the **whole** halo is rebuilt in a single
   final pass: positions and gradients are recomputed from the *originals*
   through the composed old-to-new map, one AD pass per recorded state. The
   scoring step above mutates a working pool as the search walks the
   coordinates; this final pass is what the next window actually sees. The
   wrapped model is never re-evaluated — only the transform is differentiated.

The grid is a design choice, not an approximation of a continuous search.
Centering is a bounded, one-dimensional quantity, so the candidate set can be
*enumerated* — and because it is enumerable, every candidate can be scored in a
**single pass** over the halo with online accumulators. Each halo state is
visited once per coordinate no matter how many candidates there are, nothing has
to be stored for a second look, and there is no inner optimization loop with its
own convergence behaviour to reason about. Resolution finer than `0.1` is not
what decides how the sampler behaves. The grid size is fixed and no sampler
keyword exposes it.

All of that rewrites the reparametrization **in place**, so a multi-chain run
must not let two chains share one problem. It does not: every sampler gives chain
`i` its own `deepcopy` of the `lpdf` you pass. The two that adapt a
reparametrization at all are [`adaptive_warmup_mcmc`](@ref) and
[`cooperative_warmup_mcmc`](@ref) — [`clustered_warmup_mcmc`](@ref) has no
reparametrization hooks and does not accept `nonlinear_adapt` — and for those two
the chains adapt independently, chain `i` is the same run as that chain on its
own, and the object you constructed is never mutated. Pass
`lpdfs::AbstractArray` instead to control the per-chain problems yourself; that
method is used exactly as given, so

```julia
# independently built problems — same effect as the default, spelled out
adaptive_warmup_mcmc(rngs, [ReparametrizedProblem(build_ir(), problem, backend) for _ in rngs])

# one object every chain shares — deliberate opt-in, not the default
adaptive_warmup_mcmc(rngs, fill(lpdf, length(rngs)))
```

both do what they say.

## Constraints that bite

**Coordinate order is load-bearing across a checkpoint/resume.** A checkpoint
does not store your reparametrization; it stores the fitted `source` centerings
as a bare positional list. On resume they are zipped back onto the freshly
supplied problem's `pairs` **by position**, and the stored indices are not
consulted. So a rebuild that enumerates coordinates in a different order — a
`Dict`-driven build, a data-dependent sort — silently puts every centering on the
wrong coordinate. A different *length* is not caught gracefully either: it throws
`DimensionMismatch`, or, when the overlap collapses to a single entry, silently
overwrites every pair with that one. Build `pairs` deterministically, in the same
order and with the same length, on both sides of a resume.

**Only the scalar centerings survive a checkpoint.** Not the `target`s, not the
location/log-scale closures, not the wrapped problem. That is deliberate — a
BridgeStan model and a closure are not things you want in a `.jls` file — but it
means the `lpdf` you hand to a resume has to be rebuilt by you, exactly as you
built it the first time. Nothing checks that you did; the only validation on
resume is that the dimension matches.

**The transform is on the gradient hot path.** Every `logdensity_and_gradient`
call costs one inner gradient evaluation, one extra forward pass through the
transform, and one AD pass over it. Your location and log-scale accessors run
under AD on every one of those, so keep them cheap and type-generic — indexing,
arithmetic, `exp`/`log`; not `Float64`-annotated code, not anything that mutates.

**Three construction mistakes fail late rather than at construction.** Each builds
a perfectly valid-looking object and blows up further in:

* **An under-specified Enzyme backend**, which fails late *twice*. A bare
  `AutoEnzyme()` constructs, `logdensity` works on the result, and the first
  `logdensity_and_gradient` throws `EnzymeMutabilityException`. Add
  `function_annotation=Enzyme.Const` and it gets further — all the way to the
  first restarting window — before throwing `EnzymeRuntimeActivityError`. Both
  options are needed; see the warning at the top of this page.
* **Omitting the AD backend.** `ReparametrizedProblem(r, p)` — the two-argument
  form — stores `ad_backend === nothing`. `logdensity` works fine on that object,
  so nothing looks wrong until the first `logdensity_and_gradient`, which hands
  `nothing` to `value_and_gradient` and `MethodError`s. Always pass a backend.
* **Integer centerings.** `PartiallyCentered(1)` type-parameterizes the pair
  vector on `Int`, and the fitted `Float64` centering cannot be written back into
  it: `MethodError: Cannot convert`. This does not fire at construction or on the
  first gradient — it fires at the **end of the first restarting warm-up window**,
  the first time adaptation writes a centering back. Write
  `PartiallyCentered(1.0)`.

**The per-chain `deepcopy` descends into the wrapped problem.** It has to — the
`ReparametrizedProblem` owns the `IndexedReparametrization` that warm-up rewrites,
and nothing can copy that without copying the struct holding it. What that costs
depends on what your inner problem is made of, and the two cases differ sharply:

* **Native handles are aliased, not duplicated.** A `StanProblem` holds raw
  pointers to one `bs_model_construct`ed model; `deepcopy` copies the pointers
  verbatim, so every chain's copy addresses that same native model and no chain
  reconstructs or re-`dlopen`s anything. Only the object you built carries the
  destructor, so the copies being collected does not invalidate it. It also means
  the chains still share the model's *native* state: running them with
  `parallel=true` needs a model compiled `make_args=["STAN_THREADS=true"]`, the
  same as it always did.
* **Julia-side data is genuinely copied.** An inner problem holding a large
  read-only array pays for that array once per chain, and one that cannot be
  `deepcopy`ed at all fails here rather than in your own code. Either is a reason
  to build the `lpdfs` vector yourself and share the parts you know are safe to
  share.

**`cooperative_warmup_mcmc` accepts `progress=` and drops it.** The keyword is on
the signature and passes keyword validation, but the top-level function never
forwards it to the chains it builds, so passing it has no effect. (The per-chain
constructor `WarmupHMC.cooperative_chain` does honour it.)

## See also

* [`ReparametrizedProblem`](@ref) — the wrapper, and the gradient contract.
* [`IndexedReparametrization`](@ref) — the container, and what mutates in place.
* [`Reparametrization`](@ref) — one coordinate's rule; `target` vs `source`.
* [`PartiallyCentered`](@ref) — the centering itself, and how it is fitted.
