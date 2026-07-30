using Documenter, DocumenterVitepress, WarmupHMC

# Note: the theme files under `src/.vitepress/theme/` (`htmxo-embed.ts`,
# `htmxo-gallery.css`) are mirrored from HTMXObjects.jl. Previously
# `make.jl` called `HTMXObjects.vitepress_theme_install(...)` here to
# auto-sync them. CI now clones HTMXObjects.jl to generate the static web
# recording, but the theme remains an explicit versioned snapshot instead of
# changing as a side effect of every docs build. Re-sync it manually when
# HTMXObjects ships a new embed runtime.

# WHY `repo` IS SET EXPLICITLY, AND NOT LEFT TO ORIGIN
#
# Documenter resolves the repository from the checkout's `origin` remote when
# `repo` is unset. The `repo` on `MarkdownVitepress` below does NOT satisfy that
# — it configures the Vitepress theme, not `makedocs` — so a checkout with no
# `origin` failed before rendering anything:
#
#     ArgumentError: Unable to automatically determine remote for main repo.
#     > `repo` is not set, and the Git repository has invalid origin.
#
# That is not a corner case here: the KB-managed implementation worktrees agents
# work in are credential-free and have ZERO configured remotes, so on those the
# docs build could not run AT ALL. The visible cost was an agent verifying a
# docs change by running the `@eval` fences by hand instead, because the real
# build was believed unavailable on that host — a strictly weaker check that
# cannot see `@example` blocks, `@ref` resolution or `checkdocs` (2026-07-28,
# `WarmupHMC:reparam-bench`). Setting it here makes `julia --project=docs
# docs/make.jl` work in any git checkout, remote or not.
#
# WHAT IT DOES NOT COVER: a tree that is not a git repository AT ALL. Documenter
# treats "no origin" and "no repo" as different cases, and `repo` only answers
# the first:
#
#     ArgumentError: Unable to automatically determine remote for main repo.
#     > `repo` is set but makedocs is not in a Git repository. You should
#     > configure `remotes` instead, [...]
#
# That is exactly what a `git archive` export or a release tarball is — so the
# natural way to reproduce CI's Docs job locally (archive the commit, clone the
# six dependencies beside it, run the workflow's steps) fails for a reason that
# has nothing to do with the commit under test. `actions/checkout@v4` hands CI a
# real repository, so `git init && git commit` in the reproduction tree restores
# fidelity; reach for that rather than for `remotes` (2026-07-30, verifying the
# 16-model tranche on the 1.12.6 toolchain).
makedocs(
    sitename = "WarmupHMC.jl",
    modules  = [WarmupHMC],
    repo     = Documenter.Remotes.GitHub("nsiccha", "WarmupHMC.jl"),
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/WarmupHMC.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"            => "index.md",
        "Sampler comparison" => "sampler-comparison.md",
        "Reparametrization" => "reparametrization.md",
        "Linear restart evidence" => "linear-restart.md",
        "Nonlinear weighting evidence" => "nonlinear-weighting.md",
        "Adaptive centering at fixed c" => "adaptive-centering.md",
        "BRM generated posteriors" => "brm-catalogue.md",
        "Gallery"         => "gallery.md",
        "Evidence"        => "evidence.md",
        "API"             => "api.md",
    ],
    # Every docstring on an exported name must appear in the manual, and every
    # warning is an error. `api.md` carries the exported surface explicitly plus
    # an `@autodocs Public = false` block for the internals the exported
    # docstrings cross-reference.
    checkdocs = :exports,
    warnonly = false,
)

# Ensure a root index.html redirect exists for when no stable version is deployed
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/WarmupHMC.jl",
    devbranch = "dev",
    push_preview = true,
)
