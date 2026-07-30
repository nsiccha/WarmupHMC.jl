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

"""A recorded boolean flag, as one of FOUR states rather than two.

`true` / `false` / `nothing` (recorded JSON `null`) / `missing` (the key is not
in the file at all). Both readers of `worktree_dirty` used to collapse the last
three with `get(raw, k, false) === true`, so an artifact whose harness could not
reach git read exactly like one that checked and found the tree clean. Those are
different claims and only one of them is evidence."""
function recorded_flag(text, key)
    m = match(Regex("\"$key\"\\s*:\\s*(true|false|null)"), text)
    m === nothing && return missing
    m.captures[1] == "true" ? true : m.captures[1] == "false" ? false : nothing
end

"""The narrow `src/`-only dirty flag under either of the two names harnesses
record it as — see `dirty_reason` for why there are two. `missing` only when
neither is present, so an absent unqualified key never masks a recorded
qualified one."""
function recorded_src_flag(text)
    for key in ("src_dirty", "warmuphmc_src_dirty")
        flag = recorded_flag(text, key)
        flag === missing || return flag
    end
    missing
end

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

"""Why an artifact's tree cannot be trusted to be the one its SHA names, or
`nothing` when it can.

THE PROBLEM THIS SOLVES. A recorded SHA is checkable against any later tip; a
tree with uncommitted changes is checkable against nothing, because there is no
revision to name. `code_identical.jl` — the whole engine of this script —
structurally cannot answer it. So this is the one defect here with no
retroactive remedy: the only discharge is to measure again.

WHY `src_dirty` AND NOT `worktree_dirty`. The drivers' `worktree_dirty` is
whole-tree, and it is the right thing for a human reading a provenance header.
It is the wrong thing to GATE on, because this check's subject is `src/` alone.
A dirty `docs/` cannot change what sampler ran — and a session writing up the
measurement it just took has a dirty `docs/` essentially always, so gating on
the whole tree would fire constantly for a reason that is never the reason. A
gate red for the wrong reason is a gate that gets muted rather than fixed.

WHY OLD ARTIFACTS DO NOT NEED RE-MEASURING. A clean whole tree implies a clean
`src/`, so `worktree_dirty: false` is STRICTLY STRONGER than what is being
asked and is accepted on its own. That is what makes adding the narrower flag
cheap: it costs nothing already recorded.

WHY `null` IS NOT `false`. All the helpers `catch` into `missing`, which
serialises as JSON `null` — git was unreachable when the run was recorded.
Nobody can verify anything about that tree, which is not the same claim as "it
was clean", and defaulting it to clean is the same silent-reassurance failure as
a `git fetch` that no-ops and exits 0.

WHY THE FLAG HAS TWO SPELLINGS. A harness measuring only this repo can call the
flag `src_dirty` unambiguously. The BRM harnesses cannot: they record a base for
WarmupHMC, for BayesianRegressionModels and for StanBlocks in one header, so
every field there is repo-qualified and the one this check wants is
`warmuphmc_src_dirty`. Both spellings are read, unqualified first. This is a
naming mismatch, not a missing measurement — before it was read, all four BRM
artifacts came back UNATTRIBUTABLE while sitting on a recorded
`warmuphmc_src_dirty: false`, which is exactly the "red for a reason that is
never the reason" failure the paragraph above is about."""
function dirty_reason(path)
    text = read(joinpath(REPO, path), String)
    src, whole = recorded_src_flag(text), recorded_flag(text, "worktree_dirty")

    src === true && return "`src/` had UNCOMMITTED CHANGES when this was measured, so " *
        "the base above does not name the code that ran — and no revision does. " *
        "Not dischargeable by any later check: re-measure from a clean tree"
    src === false && return nothing
    src === nothing && return "the recorded `src_dirty` is null — the harness asked git and could not " *
        "find out, so nothing about that tree is verifiable. Not the same as clean; " *
        "re-measure, or record why git was unreachable"

    # `src_dirty` absent: fall back to the whole-tree flag, which is stronger
    # where it says clean and useless where it does not.
    whole === false && return nothing
    whole === true && return "the WHOLE TREE was dirty and `src_dirty` was not recorded, so " *
        "whether `src/` was among the changes is unknowable now. Re-measure (the harness " *
        "records `src_dirty` as of this commit), or mark the run SUPERSEDED"
    whole === nothing && return "`worktree_dirty` is null and `src_dirty` is absent — git was " *
        "unreachable at measurement time and nothing narrower was recorded"
    return "NEITHER `src_dirty`/`warmuphmc_src_dirty` NOR `worktree_dirty` was recorded, " *
        "so this artifact's tree " *
        "is unverifiable. Absent is indistinguishable from clean, which is why it cannot " *
        "be read as clean; re-run the harness (both flags are recorded as of this commit)"
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

        # Checked BEFORE the base is resolved, because it outranks the result.
        # A dirty tree does not make the currency comparison fail — it makes it
        # MEANINGLESS: `code_identical.jl` would compare the recorded revision
        # against the tip and answer confidently about code that is not what
        # ran. Reporting `current` first and the dirt second would bury the
        # stronger fact under the weaker one.
        dirt = dirty_reason(p)
        if dirt !== nothing
            push!(red, p)
            println(rpad(p, 56), "  UNATTRIBUTABLE — ", dirt)
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
