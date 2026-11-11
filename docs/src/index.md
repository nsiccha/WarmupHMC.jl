# WarmupHMC.jl

WarmupHMC adapts NUTS with learned linear transformations and step sizes, plus optional partial centering for hierarchical models.

```sh
git clone --branch dev https://github.com/nsiccha/WarmupHMC.jl
julia WarmupHMC.jl/scripts/install.jl
```

Sample a standard Gaussian with a supplied gradient:

```@eval
using WarmupHMC
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl"))
ManualCache.program("gaussian")
```

```@example start
using WarmupHMC # hide
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl")) # hide
Tuple(ManualCache.load_summary()["shape"]) # hide
```

The result has one coordinate per row and one draw per column. This program's
output is cached with a checked receipt; the build does not repeat the
fit. [Gaussian calibration](examples/gaussian.md) shows its moments, diagnostics,
figure, and reproduction commands.

## Routes

| Task | Page |
| --- | --- |
| Choose an initializer and run chains | [Sampling](guides/sampling.md) |
| Continue a saved run | [Checkpoints](guides/checkpoints.md) |
| Specify a hierarchical transformation | [Reparametrization](guides/reparametrization.md) |
| Advance a chain under a changing target | [Changing targets](guides/changing-targets.md) |
| Look up arguments and return values | [Reference](reference/sampling.md) |
