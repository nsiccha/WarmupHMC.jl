# Optional sampler web application

This application is separate from the manual. The documentation build does not
start or record it. Its benchmark routes read locally regenerated results under
`bench/research/results/`, or the `WHMC_BENCH_OUT` override; a fresh checkout has
an empty benchmark index.

## Environments

Resolve `web/` and `web/app/` as separate consumer environments. The repository's
`.github/workflows/test.yml` records the public dependency branches and the
batched `Pkg.develop` setup used for the web checks. Supply the package checkout
and each unregistered dependency together, then instantiate the chosen project.
A bare instantiate cannot discover unregistered dependencies from an absent
Manifest on Julia versions that ignore `[sources]`.

The old committed Manifests contained machine-local repository paths and a
revision of a dependency that was not published at its public remote. They have
been removed. Generate deployment lockfiles in the deployment environment and
keep those machine-specific files out of the public repository. Public releases
must use published dependency revisions and portable source URLs.
