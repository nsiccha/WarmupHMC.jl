# Reparametrization

Specify a transformation from sampler coordinates to model coordinates, including
the direction and log-scale of a partial centering.

## Coordinate map

`Reparametrization(target, source, location, log_scale)` maps source coordinates
used by the sampler into target coordinates expected by the model.
`PartiallyCentered(1.0)` is centered; `PartiallyCentered(0.0)` is non-centered.
The location and log-scale can be constants or callables on the full source
position vector. Pass the logarithm of the scale, not the scale itself.

`IndexedReparametrization` attaches maps to coordinate indices.
`ReparametrizedProblem` combines that map with a LogDensityProblems target. The
target endpoint stays fixed; adaptation changes the source endpoint. Equal
endpoints give the identity initially, even if adaptation will later move them.

This construction records a non-centered sampler coordinate for a centered
model coordinate with a fixed scale. Inspecting the object verifies the direction
without running a sampler:

```@example coordinates
using WarmupHMC
transform = Reparametrization(
    PartiallyCentered(1.0), PartiallyCentered(0.0), 0.0, log(2.0))
indexed = IndexedReparametrization([1 => transform])
(; target=transform.target.c, source=transform.source.c,
   location=transform.args[1], log_scale=transform.args[2])
```

For a hierarchical model, replace the constants with accessors such as a
function reading the location coordinate or the log-scale coordinate. Refer to
the source position supplied to the accessor; do not assume another pair has
already transformed a coordinate in place. Use floating-point centering values
when their source endpoints will be adapted.

## Accessor types

All pairs must share one concrete rule type. A callable struct can represent
either a coordinate lookup or a constant without mixing closure types:

```@example coordinates
struct Coordinate
    index::Int
    value::Float64
end
(c::Coordinate)(x) = c.index == 0 ? c.value : x[c.index]
locations = [Coordinate(3, 0.0), Coordinate(0, 0.5)]
maps = IndexedReparametrization([
    i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0),
        locations[i], Coordinate(0, log(2.0))) for i in 1:2
])
(; concrete_rules=isconcretetype(eltype(maps.pairs)),
   locations=[c([0.0, 0.0, 1.5]) for c in locations])
```

The output checks both the retained type and the two accessor behaviours. Keep
the pair order and accessor meanings unchanged when rebuilding a checkpoint's
target.

## Shared locations

Many coordinates often take their location or log-scale from one computation:
a mean, or a vector of locations a model computes in one pass. Reuse one
accessor object across their pairs, or read element `i` of one function with
`SharedAccessor(f, i)`. Consecutive pairs in increasing index order that share
an accessor this way evaluate it once per vector: in the map and its inverse,
the gradient, the transport of retained states and centering selection. A block
of N coordinates then costs O(N) per evaluation rather than O(N²). The shared
function must not read the coordinates of its own block.

```@example coordinates
cell_locations(x) = x[1] .+ 0.5 .* x[2:4]   # three locations from one evaluation
cells = IndexedReparametrization([
    4 + i => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(1.0),
        SharedAccessor(cell_locations, i), 0.0) for i in 1:3
])
x = collect(1.0:7.0)
(; locations=cell_locations(x), sampler=x[5:7], model=last(cells(x))[5:7])
```

The sampler coordinates are centred and the model's are not, so each model
coordinate is its sampler coordinate minus its location.

A location and a log-scale that read one function share its evaluation, so a
model can compute both in one pass. Read the locations with
`SharedAccessor(f, i)` and the log-scale with `SharedAccessor(f, n + 1)`, for an
`f(x)` that returns `[locations; log_scale]`:

```@example coordinates
endpoints(x) = [x[1] .+ 0.5 .* x[2:4]; x[1] / 2]   # locations and log-scale at once
joint = IndexedReparametrization([
    4 + i => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(1.0),
        SharedAccessor(endpoints, i), SharedAccessor(endpoints, 4)) for i in 1:3
])
(; endpoints=endpoints(x), model=last(joint(x))[5:7])
```

Each model coordinate is now its sampler coordinate minus its location, scaled
by `exp(-0.5)`. Several blocks whose pairs all read one such function form one
run and evaluate it once; blocks that read a function each evaluate work they
share, such as a common log-scale, once per block.

A pair whose source and target centerings are equal, as every pair built with
equal ends is until warm-up moves it, maps its coordinate to itself and
evaluates neither accessor on the map, its inverse, the gradient or the
transport. Centering selection still evaluates them for its candidates.

Registered Enzyme releases, through at least 0.13.219, reject one shape of
shared accessor in static Reverse mode with `EnzymeRuntimeActivityError`: one
that can return a constant it stores, such as the `Coordinate` struct above when
consecutive pairs share it. Accessors that only read the position, as in the
example above, are not affected. The limitation is tracked upstream as
[EnzymeAD/Enzyme.jl#4045](https://github.com/EnzymeAD/Enzyme.jl/issues/4045).

## Derivatives

A sampling wrapper needs an AD backend in its third constructor argument:
`ReparametrizedProblem(indexed, problem, backend)`. The wrapper reuses the inner
problem's gradient and differentiates the coordinate map and Jacobian. Load the
package implementing the chosen DifferentiationInterface backend as well as the
interface itself. For Enzyme, load `Enzyme` and pass `AutoEnzyme()`; a function
annotation is not required.

Registered Enzyme releases compile gradients that are not memory-safe while
another thread of the process allocates, which chains running at once do for
one another. So multi-chain `adaptive_warmup_mcmc` and `resume_warmup_mcmc` with
`parallel=true`, `completion_warmup_mcmc`, `cooperative_warmup_mcmc` with
`n_cores > 1`, and multi-chain `stream_mcmc` refuse, on more than one thread and
before any work, to run chains whose wrapper WarmupHMC differentiates with such
an Enzyme. The repair is on the `dogfood` branch of
[nsiccha/Enzyme.jl](https://github.com/nsiccha/Enzyme.jl/tree/dogfood) from
`72d5d25`.

The two-argument constructor can evaluate a transformed log density, but it
does not supply a sampling gradient. An empty indexed map adapts no coordinates.
`nonlinear_adapt=true` does not discover a hierarchy in a plain target: the
coordinate map must be supplied explicitly.

## Adaptation and results

`nonlinear_adapt=false` freezes the source centering. It does not change the
coordinate convention of the returned draws: `posterior_position` is in the
wrapped model's frame for either setting. The other retained working arrays
are described in [Sampling](sampling.md).

At a restarting window, a change of centering moves the retained states and
the chain's current point into the new coordinates. If any of them is not
finite there, the change is rolled back, and the run continues exactly as if
the selection had kept the previous centering. In `adaptive_warmup_mcmc` the
warning names what was not finite: the retained states, or the current point's
coordinates, log density or gradient, with up to five coordinates and the
location and log-scale their maps read there. A pair with equal centerings
reads neither, so a location or log-scale that is not finite at the current
point shows only once selection moves its pair.

The default candidate rule can be replaced with a `CandidateScoringPlan` or a
`WindowSelectionPlan`. These advanced callbacks have different evidence and
checkpoint contracts; their complete interfaces belong in the
[transformation reference](../reference/transformations.md).

The [Gaussian calibration example](../examples/gaussian.md) evaluates a
transformed density and checks its coordinate map and Jacobian.
