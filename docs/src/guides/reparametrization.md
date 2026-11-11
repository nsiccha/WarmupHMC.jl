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

## Derivatives

A sampling wrapper needs an AD backend in its third constructor argument:
`ReparametrizedProblem(indexed, problem, backend)`. The wrapper reuses the inner
problem's gradient and differentiates the coordinate map and Jacobian. Load the
package implementing the chosen DifferentiationInterface backend as well as the
interface itself. For Enzyme, load `Enzyme` and pass `AutoEnzyme()`; a function
annotation is not required.

The two-argument constructor can evaluate a transformed log density, but it
does not supply a sampling gradient. An empty indexed map adapts no coordinates.
`nonlinear_adapt=true` does not discover a hierarchy in a plain target: the
coordinate map must be supplied explicitly.

## Adaptation and results

`nonlinear_adapt=false` freezes the source centering. It does not change the
coordinate convention of the returned draws: `posterior_position` is in the
wrapped model's frame for either setting. The other retained working arrays
are described in [Sampling](sampling.md).

The default candidate rule can be replaced with a `CandidateScoringPlan` or a
`WindowSelectionPlan`. These advanced callbacks have different evidence and
checkpoint contracts; their complete interfaces belong in the
[transformation reference](../reference/transformations.md).

The [Gaussian calibration example](../examples/gaussian.md) explains why its
target needs no nonlinear map and checks the returned model coordinates.
