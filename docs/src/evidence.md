# Benchmark evidence

Every measurement cited anywhere in this documentation comes from a JSON file
under [`docs/benchmark/results/`](https://github.com/nsiccha/WarmupHMC.jl/tree/main/docs/benchmark/results).
This page renders all of them, straight from those files, at build time.

It exists to be checkable. A figure quoted in prose is worth what the reader can
do with it, and what a reader most often wants is the surrounding context: which
revision it was measured on, what the other arms did, whether the run that
produced it errored. Those are here — the prose pages show the comparison that
makes a point, this shows the file behind it.

Two things to know before citing a row:

  * **Check the revision.** Each file records the `warmuphmc_sha` it was measured
    on, and this directory deliberately keeps superseded runs beside current
    ones — a before/after pair is usually the whole point, so the presence of a
    file is not a claim that it is current.
  * **Raw run logs are not inlined.** A file holding hundreds of per-run records
    is listed with its shape and left in the repository rather than truncated
    into an unreadable prefix. The derived views are on the pages that cite them,
    computed from these same rows by the benchmark's own code.

Nothing on this page is typed by hand. Adding a results file adds a section
here. Deletion is caught only where a page *names* the file: removing one that
prose cites fails the documentation build, while removing one that only this
page lists takes its section away silently and the build stays green. So this
appendix is a faithful view of what is checked in — not a guarantee that
everything once checked in is still there.

```@eval
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "evidence.jl"))
md_evidence_all()
```
