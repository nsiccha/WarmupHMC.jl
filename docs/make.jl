using Documenter, DocumenterVitepress, WarmupHMC

include("lint.jl")
lint_docs()

makedocs(
    sitename = "WarmupHMC.jl",
    modules = [WarmupHMC],
    repo = Documenter.Remotes.GitHub("nsiccha", "WarmupHMC.jl"),
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/WarmupHMC.jl", devurl = "dev", devbranch = "dev",
    ),
    pages = [
        "Start" => ["WarmupHMC" => "index.md"],
        "Guides" => [
            "Sampling" => "guides/sampling.md",
            "Checkpoints" => "guides/checkpoints.md",
            "Reparametrization" => "guides/reparametrization.md",
            "Changing targets" => "guides/changing-targets.md",
        ],
        "Examples" => [
            "Gaussian calibration" => "examples/gaussian.md",
            "Changing Gaussian target" => "examples/changing-target.md",
        ],
        "Reference" => [
            "Adaptive sampling" => "reference/sampling.md",
            "Streams and transitions" => "reference/streams.md",
            "Transformations" => "reference/transformations.md",
            "Compatibility" => "reference/compatibility.md",
        ],
    ],
    checkdocs = :exports,
    warnonly = false,
)

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/WarmupHMC.jl", devbranch = "dev", push_preview = true,
)
