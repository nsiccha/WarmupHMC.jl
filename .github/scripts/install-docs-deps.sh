#!/usr/bin/env bash
# The ONE list of unregistered packages the documentation build needs, shared
# by Docs.yml (build and deploy). All local packages resolve in one batch.
#
# Clones go into the checkout (CWD), as Docs.yml always did; the develops are
# batched per environment because each resolve must see every local path at
# once (ci §5).
set -euo pipefail
git clone --branch pre-inference https://github.com/nsiccha/DynamicObjects.jl.git
git clone --branch dev https://github.com/nsiccha/HTMX.jl.git
git clone --branch devibe https://github.com/nsiccha/HTMXObjects.jl.git
git clone --branch dev https://github.com/nsiccha/AlgebraOfVega.jl.git
# Treebars' integration branch `main` (decision 09eterf); `dev` diverged and
# `perf/step5-typed-slots` is frozen.
git clone --branch main https://github.com/nsiccha/Treebars.jl.git
git clone https://github.com/nsiccha/ReparametrizableDistributions.jl.git
rd_revision=$(julia --startup-file=no -e 'using TOML; print(TOML.parsefile("Project.toml")["sources"]["ReparametrizableDistributions"]["rev"])')
git -C ReparametrizableDistributions.jl checkout --detach "$rd_revision"
julia --project=docs -e 'using Pkg; Pkg.develop([
  PackageSpec(path=pwd()),
  PackageSpec(path="DynamicObjects.jl"),
  PackageSpec(path="HTMX.jl"),
  PackageSpec(path="HTMXObjects.jl"),
  PackageSpec(path="AlgebraOfVega.jl"),
  PackageSpec(path="Treebars.jl"),
  PackageSpec(path="ReparametrizableDistributions.jl"),
]); Pkg.instantiate()'
