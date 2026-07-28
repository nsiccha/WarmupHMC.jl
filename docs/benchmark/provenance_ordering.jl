# Does any harness read its git provenance INSIDE the block that truncates its
# own output file?
#
#   julia docs/benchmark/provenance_ordering.jl
#
# Green: every write-open block is free of provenance calls.
# Red:   names the file, the callee, and a line inside the offending block.
#        That line is the last one Julia recorded before the call, which for a
#        call nested inside a `JSON.print(io, Dict(...))` argument list is the
#        `JSON.print` rather than the provenance call itself — the parser emits
#        no line node inside an argument list. It lands you in the right block,
#        which is all it is for.
#
# WHAT THE DEFECT IS
#
# `open(path, "w")` truncates at open. So the natural shape
#
#     open(path, "w") do io
#         JSON.print(io, Dict(..., git_provenance()..., ...))
#     end
#
# asks git whether the tree is clean at the one instant the harness has
# guaranteed it is not: its own output file is sitting there at zero bytes.
# Once `path` is TRACKED, the artifact records `worktree_dirty = true` over a
# measurement taken from a clean tree. The trap is documented at length on
# `git_provenance` in common.jl; this is the part that can go red.
#
# WHY IT NEEDS A CHECK OF ITS OWN
#
# Nothing else in the repo can see it, and it fails toward a strong FALSE
# claim rather than a missing one. `artifact_currency.jl` compares `src/`
# between two revisions and is silent about cleanliness; `quoted_figures.jl`
# recomputes figures and never reads a provenance flag; `code_identical.jl`
# has no dirty-tree revision to compare against, by construction. The rendered
# caption then tells a reader the numbers describe code that never ran.
#
# It is also INVISIBLE UNTIL THE SECOND RUN. The first write creates an
# UNTRACKED file, which `--untracked-files=no` deliberately ignores, so a new
# harness looks correct and starts lying only once its artifact is committed —
# by which time the diff that would have shown you the bug is long landed.
#
# The pair `(worktree_dirty = true, src_dirty = false)` is its signature: the
# shape dirties the results path and can never dirty `src/`. That is a
# detector on the ARTIFACTS, and `docs/tables.jl:provenance` reads it to
# narrow the caption. It is not a detector on the SOURCE, because a harness
# whose artifact is not yet committed produces the pair on nobody's disk. This
# check is the source-side half, and the two do not overlap.
#
# MEASURED, NOT REASONED. In a throwaway clone, against a tracked artifact:
# `worktree_dirty` is `false` immediately before the `open` and `true` from
# inside it, with `src_dirty` `false` throughout. That is the whole defect,
# reproduced in three lines.
#
# WHY AN AST WALK AND NOT A GREP
#
# A grep for `git_provenance` near `open` cannot tell the fixed shape from the
# broken one — the fix leaves BOTH tokens in the file, four lines apart, and
# the correct version has the call textually FIRST. Only the block structure
# distinguishes them, so this parses.
#
# `Base.JuliaSyntax.parseall(Expr, ...)` rather than `Meta.parseall`: the
# latter does not raise on a syntax error, it embeds an `Expr(:error)` and
# returns normally, so a file this check could not parse would score clean.
#
# SCOPE, STATED RATHER THAN IMPLIED
#
# This finds the `open(path, "w") do io ... end` form, which is what every
# harness here uses. A hand-managed `io = open(path, "w")` / `close(io)` pair
# has the same defect and is NOT caught: there is no block to look inside, and
# guessing at a lexical range would redden correct code. If a harness ever
# switches to that form, extend this — do not assume it is covered.
#
# WHAT COUNTS AS A PROVENANCE CALL
#
# `git_provenance` (common.jl, the shared one) plus any callee named `dirty`
# or ending in `_dirty`, which catches the two runners that deliberately do
# not include common.jl and carry their own copy (`ace_dirty`,
# `repo_dirty`). Their duplication is documented and intentional; the ordering
# rule binds them all the same, so the check must not key on the shared name
# alone.

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# Directories holding harnesses that write artifacts into the repo. Both are
# walked recursively; a new harness anywhere under them is picked up without
# editing this list, which is the point — a census here would go stale exactly
# when someone adds the file that needed checking.
const SEARCH_DIRS = ["docs/benchmark", "bench"]

is_provenance_call(name::Symbol) =
    name === :git_provenance || name === :dirty || endswith(String(name), "_dirty")
is_provenance_call(_) = false

callee(e) = e isa Expr && e.head === :call && !isempty(e.args) ? e.args[1] : nothing

"""
Is `e` an `open(..., "w") do ... end`? Matches on the `"w"` literal appearing
among the call's arguments, so `open(p, "w")` and `open(p, "w"; kw...)` both
count, and a read-mode `open` does not.
"""
function is_write_open_do(e)
    e isa Expr && e.head === :do || return false
    call = e.args[1]
    callee(call) === :open || return false
    any(a -> a isa String && occursin('w', a), call.args[2:end])
end

"""
Walk `e`, tracking the source line and whether we are inside a write-open
block. Every provenance call found while inside one is pushed to `found`.
"""
function walk!(found, e, file; line = 0, inside = 0)
    e isa Expr || return line
    if is_write_open_do(e)
        # The `open` call itself is evaluated BEFORE the block, so its own
        # arguments are not inside it -- only the do-block body is.
        line = walk!(found, e.args[1], file; line = line, inside = inside)
        return walk!(found, e.args[2], file; line = line, inside = inside + 1, )
    end
    if inside > 0 && is_provenance_call(callee(e))
        push!(found, (file = file, line = line, call = String(callee(e))))
    end
    for a in e.args
        if a isa LineNumberNode
            line = a.line
        elseif a isa Expr
            line = walk!(found, a, file; line = line, inside = inside)
        end
    end
    line
end

function main_provenance_ordering()
    files = String[]
    for d in SEARCH_DIRS
        dir = joinpath(REPO, d)
        isdir(dir) || continue
        for (root, _, fs) in walkdir(dir), f in fs
            endswith(f, ".jl") && push!(files, joinpath(root, f))
        end
    end
    sort!(files)

    found = NamedTuple[]
    parsed = 0
    for f in files
        rel = relpath(f, REPO)
        ast = try
            Base.JuliaSyntax.parseall(Expr, read(f, String); filename = rel)
        catch err
            println("UNPARSEABLE  ", rel, "  ", sprint(showerror, err))
            push!(found, (file = rel, line = 0, call = "<unparseable>"))
            continue
        end
        parsed += 1
        walk!(found, ast, rel)
    end

    println("scanned ", parsed, " Julia file(s) under ", join(SEARCH_DIRS, ", "))
    println()

    # Guard the guard. A path typo or a moved directory would otherwise leave
    # this reporting green over nothing at all, which is the failure mode the
    # whole artifact-currency workflow is arranged around.
    if parsed == 0
        println("FAILED: zero files were parsed.")
        println("A check that inspects nothing must not report green.")
        return 1
    end

    if isempty(found)
        println("ORDERING OK: no harness reads provenance inside its own write block.")
        return 0
    end

    println("PROVENANCE READ INSIDE A WRITE BLOCK — ", length(found), " site(s):")
    for h in found
        println("  ", h.file, ":", h.line, "  ", h.call, "(...)")
    end
    println()
    println("Each of these records `worktree_dirty = true` over a clean tree as soon as")
    println("its output file is tracked. Bind the call to a local BEFORE the `open`")
    println("(`const PROV = git_provenance()`) and splat that inside the block.")
    println("Do NOT edit the recorded flag in the artifact to compensate; regenerate it")
    println("after fixing the ordering, or leave it and say why.")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_provenance_ordering())
end
