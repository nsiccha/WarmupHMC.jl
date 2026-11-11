# WarmupHMC.jl

Adaptive Hamiltonian Monte Carlo with learned linear transformations, step sizes,
and optional partial centering for hierarchical models.

[Start here](https://nsiccha.github.io/WarmupHMC.jl/dev/) for installation and a
complete sampling example. Targets implement the LogDensityProblems.jl gradient
interface; posterior draws are columns of `result.posterior_position`.

| Task | Documentation |
| --- | --- |
| Run independent chains | [Sampling](https://nsiccha.github.io/WarmupHMC.jl/dev/guides/sampling) |
| Save and resume a run | [Checkpoints](https://nsiccha.github.io/WarmupHMC.jl/dev/guides/checkpoints) |
| Specify partial centering | [Reparametrization](https://nsiccha.github.io/WarmupHMC.jl/dev/guides/reparametrization) |
| Check a known target | [Gaussian calibration](https://nsiccha.github.io/WarmupHMC.jl/dev/examples/gaussian) |
| Find an exported function | [Reference](https://nsiccha.github.io/WarmupHMC.jl/dev/reference/sampling) |

Build instructions are in [docs/README.md](docs/README.md).
Research scripts are described in [bench/README.md](bench/README.md).
