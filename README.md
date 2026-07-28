# WarmupHMC.jl

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/WarmupHMC.jl/dev/)

<!--
NO COUNT of the exports is given below, on purpose, and none should be added.

This paragraph used to read "Exports `adaptive_warmup_mcmc` plus 4
reparametrization types". That was true when it was written and was the whole
surface; three more exports arrived afterwards and it never became a merge
conflict, because a README cannot fail. A count in prose is an enumerating
consumer — it silently returns a subset, and the subset still reads as a
complete answer.

The list below IS the count. It has to agree with exactly one line:

    grep -A2 '^export' src/WarmupHMC.jl

Adding an export is therefore a two-file change. That is the cost of naming
them here at all, and it is paid deliberately: a README that points at a source
file instead of saying what you get is not much of a README.
-->

Exports the samplers `adaptive_warmup_mcmc`, `cooperative_warmup_mcmc` and
`clustered_warmup_mcmc`, plus `resume_warmup_mcmc` to continue a checkpointed
run; the reparametrization types `ReparametrizedProblem`,
`IndexedReparametrization`, `PartiallyCentered` and `Reparametrization`; and
`CandidateScoringPlan` for steering candidate adaptation.

The main method takes a (set of) `rng[s]`, a problem adhering to the LogDensityProblems.jl interface, and optional keyword arguments:
```julia
adaptive_warmup_mcmc(
    rng[s], problem; 
    n_draws=1000, 
    target_acceptance_rate=.8, 
    max_tree_depth=10, 
    progress=nothing, options...
)
```

With [Treebars.jl](https://github.com/nsiccha/Treebars.jl) progress tracking, prints a progress bar which is prettier than most. 
The result is a NamedTuple, with its `posterior_position` field containing the posterior draws. 

Results should come in faster than with "standard" methods, and should often be better.

## See also

- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) — the underlying HMC implementation
- [Treebars.jl](https://github.com/nsiccha/Treebars.jl) — progress bars for tree-based samplers
