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

function load_summary()
    cache = JSON.parsefile(joinpath(ROOT, "docs/data/gaussian.json"))
    receipt = cache["receipt"]
    receipt["program_sha256"] == digest(PROGRAMS) || error("Example program changed; regenerate the manual cache outside CI")
    receipt["source_sha256"] == source_digest() || error("Sampler source changed; regenerate the manual cache outside CI")
    cache
end

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
