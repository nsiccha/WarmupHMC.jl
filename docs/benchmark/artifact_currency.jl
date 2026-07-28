# Is every checked-in measurement still code-current?
#
#   julia docs/benchmark/artifact_currency.jl [rev]      # rev defaults to HEAD
#
# Each artifact under results/ records the commit it was measured on, in
# `warmuphmc_sha`. That SHA recedes into history while the numbers stay on the
# page, and nothing recomputes the one thing a reader actually needs to know:
# does the sampler that produced these numbers still exist? This walks every
# TRACKED artifact, reads its recorded SHA, and asks code_identical.jl whether
# `src/` at that SHA defines the same methods as `src/` at `rev`.
#
# Green: every live measurement still describes the current sampler.
# Red:   names the artifact that has gone stale, so you re-measure THAT one
#        rather than re-running the whole matrix.
#
# WHY TRACKED FILES, NOT A GLOB
#
# `git ls-files` rather than `readdir`, so a derived, gitignored file is not
# audited as if it were an artifact -- results/nonlinear_weighting/ carries
# exactly one of those, regenerated on demand and deliberately uncommitted.
#
# WHY A SUPERSEDED MARKER, AND WHY IT LIVES IN THE RUN'S OWN DIRECTORY
#
# Old runs are kept on purpose (the boxed-spec base, the halo-regression
# before/after) and they are SUPPOSED to differ from the tip -- that is what
# makes them evidence. Without a way to say so, they would hold this red
# forever and the check would be turned off within a week.
#
# Saying so is a `SUPERSEDED` file inside the run's own directory, whose first
# line is the reason, printed here on every run. Not a central manifest: a
# manifest is a second copy of the directory listing, it drifts, and the
# drifted entry points at a run that no longer exists. A marker cannot name the
# wrong directory because it IS in the directory.
#
# WHY AN EMPTY CHECK IS A FAILURE
#
# If a path changes and the enumeration matches nothing, every per-artifact
# check passes vacuously and the job goes green having verified nothing -- the
# exact failure mode probe-schema.yml's seed step exists to prevent on the
# other side of this. So zero checked artifacts is an explicit red.
#
# SHALLOW CLONES
#
# `actions/checkout@v4` defaults to depth 1: the tip and nothing else, so no
# artifact's base SHA resolves. That is reported as UNRESOLVABLE (red), never
# skipped -- a currency check that silently passes because it could not look up
# any base is worse than no check. A CI job hosting this needs
# `with: fetch-depth: 0`.

include(joinpath(@__DIR__, "code_identical.jl"))

const RESULTS = joinpath("docs", "benchmark", "results")
const SHA_RE = r"\"warmuphmc_sha\"\s*:\s*\"([0-9a-fA-F]{7,40})\""

"""Tracked `*.json` under results/, as repo-relative paths."""
function artifacts()
    out = git("ls-files", "--", RESULTS)
    sort([f for f in split(strip(out), '\n') if endswith(f, ".json")])
end

"""The recorded `warmuphmc_sha`, or `nothing`. Matched by regex rather than by
parsing, deliberately: the key sits at top level in the flat probe outputs and
under `config` in the study artifacts, and a regex needs no schema to find it
at either depth. Multiple distinct values are an error rather than a
first-match, since that would mean the artifact mixes two measured bases."""
function recorded_sha(path)
    text = read(joinpath(REPO, path), String)
    found = unique(m.captures[1] for m in eachmatch(SHA_RE, text))
    isempty(found) && return nothing
    length(found) == 1 || error("$path records $(length(found)) different warmuphmc_sha values: $found")
    only(found)
end

"""Reason from the nearest `SUPERSEDED` marker at or above `path`, stopping at
results/. `nothing` when the artifact is live."""
function superseded_reason(path)
    dir = dirname(path)
    while startswith(dir, RESULTS)
        marker = joinpath(REPO, dir, "SUPERSEDED")
        isfile(marker) && return strip(first(eachline(marker)))
        dir == RESULTS && break
        dir = dirname(dir)
    end
    nothing
end

const REPR_CACHE = Dict{String,Any}()
src_reprs(rev) = get!(REPR_CACHE, rev) do
    Dict(p => code_repr(rev, p) for p in src_files(rev))
end

resolves(rev) = success(Cmd(["git", "-C", REPO, "rev-parse", "-q", "--verify", "$rev^{commit}"]))

function main_currency(args)
    tip = isempty(args) ? "HEAD" : args[1]
    resolves(tip) || error("tip revision `$tip` does not resolve")

    paths = artifacts()
    checked, red = 0, String[]

    for p in paths
        reason = superseded_reason(p)
        if reason !== nothing
            println(rpad(p, 56), "  superseded — ", reason)
            continue
        end

        sha = recorded_sha(p)
        if sha === nothing
            push!(red, p)
            println(rpad(p, 56), "  NO warmuphmc_sha — cannot be checked; record one, or mark the run SUPERSEDED")
            continue
        end
        if !resolves(sha)
            push!(red, p)
            println(rpad(p, 56), "  UNRESOLVABLE base $sha — shallow clone? CI needs `with: fetch-depth: 0`")
            continue
        end

        checked += 1
        base, head = src_reprs(sha), src_reprs(tip)
        differing = [f for f in union(keys(base), keys(head)) if get(base, f, nothing) != get(head, f, nothing)]
        if isempty(differing)
            println(rpad(p, 56), "  current — $(sha[1:7]) is code-identical to $tip")
        else
            push!(red, p)
            println(rpad(p, 56), "  STALE — $(sha[1:7]) differs from $tip in ", join(sort(differing), ", "))
        end
    end

    println()
    if checked == 0
        println("FAILED: zero artifacts were checked, over $(length(paths)) tracked file(s).")
        println("A currency check that verifies nothing must not report green.")
        return 1
    end
    if isempty(red)
        println("ALL CURRENT: $checked live artifact(s) still describe the sampler at $tip.")
        return 0
    end
    println("STALE OR UNCHECKABLE: ", length(red), " artifact(s) — ", join(red, ", "))
    println("Re-measure those, or mark the run SUPERSEDED with a reason if it is kept as history.")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_currency(ARGS))
end
