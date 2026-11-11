"""Check the manual's scope and executable-source conventions before rendering."""
function lint_docs()
    root = joinpath(@__DIR__, "src")
    make = read(joinpath(@__DIR__, "make.jl"), String)
    inventory = read(joinpath(@__DIR__, "README.md"), String)
    groups = [m.captures[1] for m in eachmatch(r"^        \"([^\"]+)\" => \["m, make)]
    groups == ["Start", "Guides", "Examples", "Reference"] || error("Manual navigation must have the four ordered groups")
    pages = sort([relpath(joinpath(dir, file), root) for (dir, _, files) in walkdir(root)
                  for file in files if endswith(file, ".md")])
    listed = sort([m.captures[1] for m in eachmatch(r"=> \"([^\"]+\.md)\"", make)])
    pages == listed || error("Manual pages differ from the navigation: $(setdiff(pages, listed)) / $(setdiff(listed, pages))")
    banned = r"\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting"i
    for page in pages
        source = read(joinpath(root, page), String)
        occursin("`$page`", inventory) || error("Page missing from docs/README.md: $page")
        occursin(r"^```(?:julia|python|r|javascript)\s*$"m, source) && error("Static runnable fence: $page")
        prose = replace(source, r"```[^\n]*\n.*?```"s => "")
        occursin(banned, prose) && error("Prohibited prose in $page")
        words = length(split(prose))
        startswith(page, "guides/") && words >= 1500 && error("Guide exceeds 1499 words: $page")
        if startswith(page, "examples/")
            800 <= words <= 1500 || error("Example must have 800–1500 prose words: $page has $words")
            occursin("```@example", source) || error("Example has no executed block: $page")
            occursin("AlgebraOfVega.to_json", source) || error("Example has no build-generated figure: $page")
            occursin("::: details Reproduce", source) || error("Example has no reproduction footer: $page")
        end
        println("docs lint: ", page, " (", words, " prose words)")
    end
end
