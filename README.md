# WarmupHMC.jl

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/WarmupHMC.jl/dev/)

Exports a single method, taking a (set of) `rng[s]`, a problem adhering to the LogDensityProblems.jl interface, and optional keyword arguments: 
```julia
adaptive_warmup_mcmc(
    rng[s], problem; 
    n_draws=1000, 
    target_acceptance_rate=.8, 
    max_tree_depth=10, 
    progress=nothing, options...
)
````

With `progress=Term.ProgressBar`, prints a progress bar which is prettier than most. 
The result is a NamedTuple, with its `posterior_position` field containing the posterior draws. 

Results should come in faster than with "standard" methods, and should often be better.

## See also

- [ReactiveHMC.jl](https://github.com/nsiccha/ReactiveHMC.jl) — underlying HMC implementation
- [ReactiveObjects.jl](https://github.com/nsiccha/ReactiveObjects.jl) — reactive algorithmic kernels
- [Treebars.jl](https://github.com/nsiccha/Treebars.jl) — progress bars for tree-based samplers