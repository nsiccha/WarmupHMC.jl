# Research drivers

This directory contains the former `docs/benchmark` generators. Its project is
independent of the documentation project. Outputs are local research products;
they are neither checked in nor rendered into the manual. Driver headers specify
commands and configuration. The optional `WHMC_BENCH_OUT` environment variable
selects an output directory. Without it the drivers write under `results/`.

`artifact_currency.jl`, `code_identical.jl`, and `provenance_ordering.jl` retain
the provenance checks for researchers who regenerate artifacts. Historical
prose-number checks and their CI workflow were removed with the historical
results catalogue. The web app handles an absent results directory as an empty
benchmark index.
