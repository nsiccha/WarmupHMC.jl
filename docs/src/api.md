# API Reference

## What semver covers

WarmupHMC follows [semantic versioning](https://semver.org). That promise is
only worth reading if it says what it is *about*, so here is the whole of it.

**Covered — a breaking change to any of these needs a major version:**

* **The exported names** listed on this page
  ([`adaptive_warmup_mcmc`](@ref), [`cooperative_warmup_mcmc`](@ref),
  [`completion_warmup_mcmc`](@ref), [`stream_mcmc`](@ref), [`open_stream`](@ref),
  [`clustered_warmup_mcmc`](@ref), [`resume_warmup_mcmc`](@ref),
  [`ReparametrizedProblem`](@ref), [`IndexedReparametrization`](@ref),
  [`Reparametrization`](@ref), [`PartiallyCentered`](@ref),
  [`CandidateScoringPlan`](@ref)) and the behaviour their docstrings describe.
* **The keywords [`adaptive_warmup_mcmc`](@ref) accepts.** Passing one it does not
  accept is an error rather than a silent no-op, which makes the accepted set
  part of the interface — so removing or renaming one breaks callers.
* **The field names of the `NamedTuple` it returns**, and their order. The
  order matters because a `NamedTuple`'s names *are* its type: reordering them
  changes the type even though every field is still there.

**Not covered:**

* Anything under `WarmupHMC.` that is not exported — see [Internals](@ref)
  below.
* The **numbers**. Draws, step sizes, evaluation counts and the adaptation path
  itself change whenever the sampler improves; that is the point of the package.
  A minor release may return different draws for the same seed.
* The **types** of the returned fields — element types, array types, and whether
  a field is materialised or lazy.
* Anything reached through a dependency's types rather than through the names
  above.

The first two covered bullets are not prose: `test/public_api.jl` holds them as
literal lists and the suite fails if the code moves away from them. That is
deliberate — the test *is* the contract, so widening or narrowing the public
surface requires editing it on purpose, which is the moment to ask whether the
change is breaking.

## Sampling

```@docs
WarmupHMC.adaptive_warmup_mcmc
WarmupHMC.cooperative_warmup_mcmc
WarmupHMC.completion_warmup_mcmc
WarmupHMC.stream_mcmc
WarmupHMC.open_stream
WarmupHMC.clustered_warmup_mcmc
WarmupHMC.resume_warmup_mcmc
```

## Reparametrizations

```@docs
WarmupHMC.ReparametrizedProblem
WarmupHMC.CandidateScoringPlan
WarmupHMC.IndexedReparametrization
WarmupHMC.PartiallyCentered
WarmupHMC.Reparametrization
```

## Internals

Not part of the public API — no compatibility is promised for anything below,
and none of it is pinned by `test/public_api.jl`. They are listed because the
exported docstrings above refer to them: a name being documented here is a
consequence of `checkdocs = :exports`, not a promise about it.

```@autodocs
Modules = [WarmupHMC]
Public = false
```
