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
run; `stream_mcmc` for interruptible fixed-kernel sampling with `open_stream` to
read its output back; the reparametrization types `ReparametrizedProblem`,
`IndexedReparametrization`, `PartiallyCentered` and `Reparametrization`; and
`CandidateScoringPlan` for steering candidate adaptation. The opt-in
`completion_warmup_mcmc` runs independent adaptive chains with a completion
quorum and grace period.

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

Once the kernel is fixed — a mass-matrix scale, a step size and a tree depth,
whether hand-supplied or read from a warm-up checkpoint — `stream_mcmc` runs pure
sampling with no adaptation and streams the draws into a zeroed, memory-mappable
file. It persists only the RNG state and the last safe draw index into a
crash-safe "safe ring" beside that file, so a process killed mid-sample or
mid-write resumes byte-for-byte from the immediately preceding state on the next
call to the same `path`. `open_stream` re-maps a finished or in-progress run for
reading.

```julia
stream_mcmc(rng, problem, position; path, n_draws, metric, stepsize)  # start
stream_mcmc(problem; path, n_draws, metric, stepsize)                  # resume the same path
stream_mcmc(checkpoint, problem; path, n_draws)                        # seed from a WarmupHMC checkpoint
stream_mcmc(checkpoint, problem, position; path, n_draws)             # checkpoint kernel, explicit start point
stream_mcmc(checkpoint, problem, positions; path, n_draws)            # N chains under path/chain_<i>, one per start
```

That sentence used to stand on its own. It is now measured: `bench/sampler_comparison.jl`
runs WarmupHMC, DynamicHMC and AdvancedHMC over the same targets, and
[WarmupHMC vs other samplers](https://nsiccha.github.io/WarmupHMC.jl/dev/sampler-comparison)
renders the result — verdict included — from the checked-in JSON at build time.
Read that page for what "faster" does and does not mean here. The lead is in
**gradient evaluations** — the portable number, and the one that reproduces:
re-running identical seeds returns bit-identical counts, so that verdict does
not move. Wall-clock is a different story, and the page says so out of its own
rows rather than in a caveat: its verdict changes between repeats of the *same*
seeds, because the timing noise is wider than the band used to call a winner.
Nothing here measures whether the draws are correct.

## Completion quorum (opt-in)

```julia
using Random, WarmupHMC
out = completion_warmup_mcmc([Xoshiro(i) for i in 1:16], problem;
    n_draws=1000, min_completed=12, grace_seconds=30,
    checkpoint_dir="completion-run")
chain_ids = [r.chain_index for r in out.results]
draw_count = out.completion.n_samples
```

The twelfth completed chain starts the grace period. All full chains admitted
before the cutoff are returned, with their original identities. Grace is a
**stop-request time**: unfinished workers stop at checkpoint boundaries and are
joined before return, so initialization or a long window can delay the return.
Warmup windows do not count as completed production draws. Chain failures and
omissions are explicit in `out.completion`; an unmet quorum throws
`WarmupHMC.CompletionQuorumError` carrying the settled `outcome`.

**Selecting faster chains can bias inference** when runtime depends on sampled
states or modes. R-hat, bulk/tail ESS and divergences describe only retained
draws and cannot rule out this bias. Report the policy, omissions and actual
returned counts. The ordinary samplers retain their existing behavior.

Checkpoints keep original `chain_<i>` identities and adaptive resume state.
An interrupted run can resume with the same ordered RNG/density slots;
already-complete checkpoints are admitted before scheduling. Reopening a
successfully finished run with `resume=true` returns its recorded outcome and
selection, even if omitted checkpoints have since advanced. Changing the
terminal draw target or policy requires a new run. Each invocation records its
policy and terminal counts under the returned `completion.attempt_directory`;
the full contract is in `?completion_warmup_mcmc`.

## Stability

Semantic versioning covers the exported names above, the keywords
`adaptive_warmup_mcmc` accepts, and the field names of the `NamedTuple` it
returns. It deliberately does **not** cover the numbers: draws, step sizes and
the adaptation path change whenever the sampler improves, so a minor release
may return different draws for the same seed. The full statement is
[What semver covers](https://nsiccha.github.io/WarmupHMC.jl/dev/api#What-semver-covers)
in the API reference, and `test/public_api.jl` enforces it.

## See also

- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) — the underlying HMC implementation
- [Treebars.jl](https://github.com/nsiccha/Treebars.jl) — progress bars for tree-based samplers
