module ManualCache

using JSON, SHA, Markdown, WarmupHMC

const ROOT = pkgdir(WarmupHMC)
const PROGRAMS = ["bench/docs/gaussian.jl", "bench/docs/checkpoint.jl"]

function source_digest()
    paths = ["Project.toml"]
    for directory in ("src", "ext"), (root, _, files) in walkdir(joinpath(ROOT, directory))
        append!(paths, [relpath(joinpath(root, file), ROOT) for file in files if endswith(file, ".jl")])
    end
    digest(sort!(paths))
end

function digest(paths)
    bytes2hex(sha256(join([path * "\n" * read(joinpath(ROOT, path), String) for path in paths], "\n")))
end

# A cached fit is keyed by its content: the displayed programs, which fix the
# target, settings and seed. A changed program means the saved output no longer
# belongs to the code on the page, so the build refuses it. The package source
# is provenance, not part of the key: ordinary sampler commits must not stop the
# manual from building, and `source_current` reports the difference instead.
function load_summary()
    cache = JSON.parsefile(joinpath(ROOT, "docs/data/gaussian.json"))
    cache["receipt"]["program_sha256"] == digest(PROGRAMS) || error("Example program changed; regenerate the manual cache outside CI")
    cache
end

"""Whether the package source this page is built from produced the cached run."""
source_current(cache) = cache["receipt"]["source_sha256"] == source_digest()

"""The displayed program is the exact content verified by the run receipt."""
function program(name)
    load_summary()
    Markdown.MD(Markdown.Code("julia", read(joinpath(ROOT, "bench/docs", name * ".jl"), String)))
end

function table(headers, rows)
    Markdown.parse(join([
        "| " * join(headers, " | ") * " |",
        "| " * join(fill("---", length(headers)), " | ") * " |",
        ["| " * join(row, " | ") * " |" for row in rows]...,
    ], "\n"))
end

end
