#!/usr/bin/env bash
# The ONE list of unregistered packages the documentation build needs, shared
# by Docs.yml (build + deploy) and probe-schema.yml (build against freshly
# regenerated probes). Two hand-copied lists drifted: Docs.yml gained
# AlgebraOfVega while probe-schema.yml kept a bare `Pkg.instantiate()` on the
# docs env, which failed on every run from 2026-07-30 with
# "expected package `AlgebraOfVega` to be registered".
#
# Clones go into the checkout (CWD), as Docs.yml always did; the develops are
# batched per environment because each resolve must see every local path at
# once (ci §5).
set -euo pipefail
git clone --branch pre-inference https://github.com/nsiccha/DynamicObjects.jl.git
git clone --branch dev https://github.com/nsiccha/HTMX.jl.git
git clone --branch devibe https://github.com/nsiccha/HTMXObjects.jl.git
git clone --branch dev https://github.com/nsiccha/AlgebraOfVega.jl.git
git clone --branch dev https://github.com/nsiccha/TestModules.jl.git
# Treebars' canonical line, not `dev` (diverged; lacks interrupt_requested).
git clone --branch perf/step5-typed-slots https://github.com/nsiccha/Treebars.jl.git
julia --project=web -e 'using Pkg; Pkg.develop([
  PackageSpec(path=pwd()),
  PackageSpec(path="DynamicObjects.jl"),
  PackageSpec(path="HTMX.jl"),
  PackageSpec(path="HTMXObjects.jl"),
  PackageSpec(path="TestModules.jl"),
  PackageSpec(path="Treebars.jl"),
]); Pkg.instantiate()'
julia --project=docs -e 'using Pkg; Pkg.develop([
  PackageSpec(path=pwd()),
  PackageSpec(path="DynamicObjects.jl"),
  PackageSpec(path="HTMX.jl"),
  PackageSpec(path="HTMXObjects.jl"),
  PackageSpec(path="AlgebraOfVega.jl"),
  PackageSpec(path="TestModules.jl"),
  PackageSpec(path="Treebars.jl"),
]); Pkg.instantiate()'
