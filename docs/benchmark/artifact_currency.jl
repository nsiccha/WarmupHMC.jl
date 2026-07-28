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
# WHEN THE BASE CANNOT BE USED
#
# A base that cannot be compared is reported red, never skipped -- a currency
# check that silently passes because it could not look up any base is worse than
# no check.
#
# But "cannot be used" is THREE states with three different fixes, and this
# script used to print one message for all of them, naming the fix for only the
# first:
#
#   * absent from a SHALLOW clone -- `actions/checkout@v4` defaults to depth 1,
#     so no base resolves. `with: fetch-depth: 0` is the fix.
#   * absent from a FULL clone -- every ref was fetched and the commit is on
#     none of them. No checkout setting helps.
#   * PRESENT but contained by zero refs -- resolvable here, resolvable nowhere
#     else, and only until the next `gc`.
#
# The last two are real in this repo: `5637fcf` and `b5c7dee` are present as
# objects and `git for-each-ref --contains` returns nothing for either. Telling
# a reader to set `fetch-depth: 0` for those sends them to fix a checkout config
# that is already correct. See `unresolvable_reason`.

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

"""Is this checkout a shallow clone? The precise discriminator for the
`fetch-depth` hint -- see `unresolvable_reason`."""
is_shallow() = strip(git("rev-parse", "--is-shallow-repository")) == "true"

"""How many refs contain `rev`. Zero means the commit is present as an object
but reachable from nothing: it survives only until the next `gc`, and no fetch
setting recovers it."""
function containing_refs(rev)
    # `String[...]`, not `[...]`: `rev` arrives as a `SubString` from the regex
    # capture, which widens the literal to `Vector{AbstractString}` and there is
    # no `Cmd` method for that.
    out = read(Cmd(String["git", "-C", REPO, "for-each-ref", "--contains", rev,
                          "--format=%(refname)"]), String)
    count(!isempty, split(strip(out), '\n'; keepempty = false))
end

"""Why a base SHA could not be used, as a message naming the fix that applies.

Three states hide behind one failed `rev-parse`, and they call for opposite
responses -- which is the whole reason this is not one branch:

  * **absent, shallow clone** -- the object was never fetched. `fetch-depth: 0`
    fixes it, and this is the only case where saying so is correct.
  * **absent, full clone** -- a full fetch already brought every ref, so the
    commit is on none of them. No checkout setting recovers it; it was rebased
    away or never published.
  * **present, on no ref** -- the object is here and the comparison below is
    computable, but it is unreachable: `git gc` may drop it at any time, and a
    fresh clone will not have it at all. The artifact's provenance is therefore
    not reproducible by anyone else, which is what the check exists to
    establish.

Reporting all three as "shallow clone?" sends a reader to fix a checkout
setting that is already correct -- and in the third case, to fix one while the
comparison it would enable is running fine."""
function unresolvable_reason(sha)
    if resolves(sha)
        return "on NO REF — present as an object but reachable from nothing, so " *
               "`git gc` may drop it and a fresh clone never had it. NOT a checkout " *
               "setting: re-measure, or record why this base is unreachable"
    end
    is_shallow() ?
        "NOT FETCHED — this is a shallow clone; CI needs `with: fetch-depth: 0`" :
        "NOT FETCHED — and this is a FULL clone, so no `fetch-depth` change helps: " *
        "the commit is on no fetched ref (rebased away, or never published)"
end

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
        usable = resolves(sha) && containing_refs(sha) > 0
        if !usable
            push!(red, p)
            println(rpad(p, 56), "  UNUSABLE base $sha — ", unresolvable_reason(sha))
            # An unreachable-but-present base can still be compared, and the
            # answer is worth printing: it separates "the provenance is not
            # reproducible" from "and the numbers are wrong too". It does NOT
            # count toward `checked` -- a base nobody else can resolve is not a
            # verified one, whatever the comparison says.
            if resolves(sha)
                base, head = src_reprs(sha), src_reprs(tip)
                differing = [f for f in union(keys(base), keys(head))
                             if get(base, f, nothing) != get(head, f, nothing)]
                println(" "^58, "(locally: ", isempty(differing) ?
                    "code-identical to $tip anyway" :
                    "also differs in " * join(sort(differing), ", "), ")")
            end
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
