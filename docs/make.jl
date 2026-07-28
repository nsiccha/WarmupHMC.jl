using Documenter, DocumenterVitepress, WarmupHMC

# Note: the theme files under `src/.vitepress/theme/` (`htmxo-embed.ts`,
# `htmxo-gallery.css`) are mirrored from HTMXObjects.jl. Previously
# `make.jl` called `HTMXObjects.vitepress_theme_install(...)` here to
# auto-sync them. CI now clones HTMXObjects.jl to generate the static web
# recording, but the theme remains an explicit versioned snapshot instead of
# changing as a side effect of every docs build. Re-sync it manually when
# HTMXObjects ships a new embed runtime.

makedocs(
    sitename = "WarmupHMC.jl",
    modules  = [WarmupHMC],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/WarmupHMC.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"            => "index.md",
        "Reparametrization" => "reparametrization.md",
        "Linear restart evidence" => "linear-restart.md",
        "Nonlinear weighting evidence" => "nonlinear-weighting.md",
        "Adaptive centering at fixed c" => "adaptive-centering.md",
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
