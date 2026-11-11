# Manual example cache

`gaussian.jl` is a complete sampler program. `checkpoint.jl` continues with the
same target and demonstrates retained-draw continuation. `run.jl` runs both,
reduces the draws to coordinate summaries, and writes `docs/data/gaussian.json`.
No draws or figures are committed.

After installing the docs environment, run:

```sh
julia --project=docs bench/docs/run.jl
julia --project=docs docs/make.jl
```

The receipt hashes the exact displayed programs and the package's `Project.toml`,
`src/`, and `ext/` files. The build refuses a mismatched cache. It also records
Julia and selected dependency versions, the base Git revision, UTC time, and the
regeneration command. The content hash identifies the executed source even when
the base revision precedes a documentation commit. The cached program is shown
with its saved output; it is not rerun in CI. Plot construction and all derived
tables run in the documentation build.

Run regeneration only when source or example inputs change. Inspect the output
before committing the small JSON summary. This is a local cache contract pending
the shared documentation fit adapter; it does not claim reproducibility across
unrecorded dependency changes.
