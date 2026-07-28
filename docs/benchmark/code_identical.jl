# Are two revisions of the sampler CODE-identical, ignoring docstrings?
#
#   julia docs/benchmark/code_identical.jl <rev-a> <rev-b> [path ...]
#
# Every table in RESULTS.md names the commit it was measured on, and that commit
# is never the tip by the time the tables land. The question a reader then has is
# whether the numbers still describe the current sampler. Answering it by reading
# `git diff` is unreliable in one specific direction: a diff that is *all prose*
# looks exactly like a diff that is *mostly prose*, and the docstrings in
# src/Reparametrizations.jl are long enough to bury a one-line code change in
# a 121-line diff. That is not hypothetical here — the file has been edited for
# prose alone three times since the base these tables were measured on.
#
# So this compares the parsed AST with docstrings and line numbers stripped.
# Byte-identical output means the two revisions define the same methods with the
# same bodies; the gradient path cannot differ. Prose is invisible to it, and a
# single changed token is not.
#
# No dependencies -- runs under plain `julia`, deliberately, so a provenance
# check never needs the benchmark environment (which drags BridgeStan and
# PosteriorDB) to answer a question about source code.
#
# Exit 0 = identical, 1 = differs, so it works as a gate. Verified in both
# directions before it was committed:
#
#   5637fcf..73ebf98  src/Reparametrizations.jl  -> IDENTICAL (three docstring commits)
#   b0a1c4f~1..b0a1c4f  same file                -> DIFFERS   (a real code change)
#
# The second is the one that matters: a check that has never been observed to
# fail is not evidence of anything.

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

git(args...) = read(`git -C $REPO $args`, String)

"""Files under `src/` in a revision, as a Set. Compared as a UNION across the
two revisions, never as one side's listing: a file that exists in only one of
them is a difference, and enumerating one side would silently skip it."""
function src_files(rev)
    out = git("ls-tree", "-r", "--name-only", rev, "src/")
    Set(f for f in split(strip(out), '\n') if endswith(f, ".jl"))
end

isdocmacro(e) = e isa Expr && e.head === :macrocall && !isempty(e.args) &&
    (e.args[1] === GlobalRef(Core, Symbol("@doc")) || e.args[1] === Symbol("@doc") ||
     string(e.args[1]) in ("@doc", "Core.@doc"))

"""Drop docstrings (replacing a `@doc` call with the thing it documents) and
every `LineNumberNode`, so moving code down a file is not reported as a change."""
function strip_docs(e)
    e isa Expr || return e
    isdocmacro(e) && return strip_docs(e.args[end])
    Expr(e.head, Any[strip_docs(a) for a in e.args if !(a isa LineNumberNode)]...)
end

"""`nothing` if the path does not exist in that revision -- distinct from an
empty file, and reported as a difference rather than skipped."""
function code_repr(rev, path)
    src = try
        git("show", "$rev:$path")
    catch
        return nothing
    end
    repr(strip_docs(Meta.parseall(src; filename=path)))
end

function main(args)
    length(args) >= 2 || error("usage: julia code_identical.jl <rev-a> <rev-b> [path ...]")
    a, b = args[1], args[2]
    paths = length(args) > 2 ? args[3:end] :
        sort(collect(union(src_files(a), src_files(b))))

    differing = String[]
    for p in paths
        ra, rb = code_repr(a, p), code_repr(b, p)
        status = if ra === nothing && rb === nothing
            "ABSENT in both"
        elseif ra === nothing
            push!(differing, p); "ADDED in $b"
        elseif rb === nothing
            push!(differing, p); "REMOVED in $b"
        elseif ra == rb
            "identical"
        else
            push!(differing, p); "DIFFERS"
        end
        println(rpad(p, 40), "  ", status)
    end

    println()
    if isempty(differing)
        println("CODE-IDENTICAL: $a and $b define the same methods across ",
                length(paths), " file(s). Differences are docstrings only.")
        return 0
    end
    println("CODE DIFFERS in ", length(differing), " file(s): ", join(differing, ", "))
    println("Benchmark tables measured on $a do not automatically describe $b.")
    return 1
end

# Guarded so `artifact_currency.jl` can `include` this for `code_repr` /
# `src_files` without the include itself calling `exit`.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
