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

## Exports

Each group links to the reference page that documents it.

- [Adaptive sampling](https://nsiccha.github.io/WarmupHMC.jl/dev/reference/sampling):
  `adaptive_warmup_mcmc`, `resume_warmup_mcmc`, `cooperative_warmup_mcmc`,
  `completion_warmup_mcmc`.
- [Streams and transitions](https://nsiccha.github.io/WarmupHMC.jl/dev/reference/streams):
  `stream_mcmc`, `open_stream`, `close_stream`, `FixedKernelNUTSState`,
  `fixed_kernel_nuts_step`, `ContinuousNUTSState`, `continuous_nuts_step`,
  `reposition!`.
- [Transformations](https://nsiccha.github.io/WarmupHMC.jl/dev/reference/transformations):
  `scale_data`, `scale_from_data`, `fit_gaussian_moments`, `PartiallyCentered`
  (re-exported from ReparametrizableDistributions), `Reparametrization`,
  `SharedAccessor`, `IndexedReparametrization`, `MarkovChain`,
  `ReparametrizedProblem`, `CandidateScoringPlan`, `candidate_scoring_losses`,
  `WindowSelectionPlan`.

Build instructions are in [docs/README.md](docs/README.md).
Research scripts are described in [bench/README.md](bench/README.md).
