# Rendering checked-in benchmark JSON as documentation tables.
#
# Documenter never executes a plain ```julia fence, so any number typed into
# one is a hand-copy that nothing can check. Numbers that came out of a
# measurement therefore go through an `@eval` block, which Documenter *does*
# run, and which reads the same JSON the benchmark harness wrote and the web
# app consumes. There is one source of truth per number, and `make.jl` fails
# the build if it is missing or malformed.
#
# Use from a page:
#
#     ```@eval
#     Base.include(@__MODULE__, joinpath(@__DIR__, "..", "tables.jl"))
#     d = load_results("annotation_sweep.json")
#     md_table(["target", "d"], [[r["target"], r["dim"]] for r in d["rows"]])
#     ```
#
# `@__DIR__` inside an `@eval` block is `docs/build`, so `joinpath(@__DIR__,
# "..")` is `docs/` in both a local and a CI build. Do not use `pwd()`.

import JSON, Markdown, Printf

"""
Directory holding the checked-in benchmark result JSON.

`WHMC_BENCH_OUT` overrides it, which is what lets a CI job point the *consumer*
at results a generator has just written. The generators under `docs/benchmark/`
already honour that variable; without it on this side there is no job that can
run a probe and `makedocs` against the same files, and a generator is free to
change its output keys next to a stale checked-in JSON indefinitely. That is not
hypothetical — `655218e` changed `capture_boxing.jl`'s keys and nothing surfaced
it until `176a061` regenerated the results and took the docs build red.

Measured, that gap was **2 h 12 m on one day**, not the months an earlier draft
of this docstring claimed. The duration was never what made it dangerous. Two
other things did, and they are what this seam is actually for: the renaming
landed in one owner's file and the breakage surfaced in another's, and
`evidence.md` — which enumerates `results/` rather than naming keys — absorbed
the new shape in silence, so only the page that NAMED a removed key went red.
An enumerating consumer cannot fail; a naming one is the whole gate.

**A blank value is an ERROR, not a fallback.** `get(ENV, k, default)` returns
`""` for an exported-but-empty variable, and `joinpath("", "x")` is *relative*,
so the naive form silently resolves to the process's working directory. On the
generator side that scatters result files into the repo root. Here it is worse
in a subtler way: falling back to the checked-in results would let a CI guard
that regenerates into a temp directory build green against the very copies it
deliberately deleted — passing while checking nothing, which is the exact state
that guard exists to prevent. Refusing blank closes it independent of step
order.

Mirrors `env_dir` in `docs/benchmark/common.jl`. Duplicated rather than shared
on purpose: `common.jl` pulls PosteriorDB, BridgeStan and
StanLogDensityProblems, none of which `docs/Project.toml` has, so including it
would drag the whole measurement stack into `makedocs`.

This is not the only such copy. `env_dir` serves every generator that can
afford `common.jl`, and the scripts that cannot afford it inline their own
guard. Change the rule and you must change all of them, so find them with:

    grep -rl 'is set but blank' docs/

**Do not anchor that search on the variable name.** One rule guards several —
`WHMC_BENCH_OUT` here and in most generators, `WHMC_NW_OUT` in
`nonlinear_weighting_run.jl`, `WHMC_LINEAR_BENCH_OUT` in
`run_linear_restart_benchmark.jl` — so a variable grep silently returns a
*subset*, which is the failure direction that matters: a site you never see is
a site you never fix.

The error string cannot fail that way. It over-reports instead — a file that
merely *cites* this anchor matches it too — and that is the direction to
prefer, because an extra file to glance at costs seconds while a missed one
ships the bug. For the same reason this paragraph gives no count: a number in
prose has nothing to check it, and every count this note has carried was wrong
within the hour of being written.
"""
function results_dir()
    haskey(ENV, "WHMC_BENCH_OUT") ||
        return normpath(joinpath(@__DIR__, "benchmark", "results"))
    override = ENV["WHMC_BENCH_OUT"]
    isempty(strip(override)) && error("""
        WHMC_BENCH_OUT is set but blank.

        A blank value is almost always an unset shell variable that expanded to
        the empty string. It is refused rather than defaulted, because falling
        back to the checked-in results would let a guard that regenerates into a
        temporary directory pass while reading the copies it meant to replace.

        Unset WHMC_BENCH_OUT to read the checked-in results, or give it a real
        path.
        """)
    normpath(override)
end

"""
    load_results(relpath) -> Dict

Parse a checked-in result file under `docs/benchmark/results/`. Throws if the
file is absent — an `@eval` block that throws fails the docs build, which is
the point: a table whose data has gone missing must not render as an empty one.
"""
function load_results(relpath::AbstractString)
    path = joinpath(results_dir(), relpath)
    isfile(path) || error("""
        Benchmark results not found: $(path)

        Documentation tables are generated from the checked-in JSON under
        docs/benchmark/results/. Either the file was not committed, or a page
        names it wrongly. Do not replace this table with typed-in numbers.
        """)
    JSON.parse(read(path, String))
end

"""
    provenance(d; harness) -> String

One sentence naming what produced the numbers, built from the result file's own
metadata rather than from prose. A result file that records no `warmuphmc_sha`
says so instead of silently rendering an unattributed table.

`harness` is the **repo-relative path** of the script that wrote the file —
`"docs/benchmark/annotation_sweep.jl"`, not `"annotation_sweep.jl"`. It used to
be a bare filename with `docs/benchmark/` prepended here, which encoded an
assumption nobody had checked: that every harness lives in that one directory.
`bench/sampler_comparison.jl` does not, and cannot — it pulls in AdvancedHMC,
which `docs/benchmark/Project.toml` must not carry. A hardcoded prefix does not
go red when its assumption breaks; it renders a plausible path to a file that
is not there, which is the same shape as a caption that names the wrong script.
Note the deliberate mismatch with [`load_harness`](@ref), whose argument stays
relative to `docs/benchmark/`: that one has to *find* a file, so its base
directory is a real constraint rather than an assumption about where authors
put things.

Two artifact shapes are supported, because a harness should not have to repeat
itself to satisfy this function. If the file has a `config` object, provenance is
read from there — the preferred shape, since a single copy of `warmuphmc_sha`
cannot disagree with a duplicate of itself. Otherwise, or for any field `config`
omits, it falls back to the top level, which is the older flat shape.

**A SHA recorded from a DIRTY worktree is not provenance at all**, and it is the
one failure here with no detector but the flag itself. `warmuphmc_sha` renders
identically whether the tree was clean or carried uncommitted changes, so it
reads as full attribution while pointing at code that never ran. Unlike a stale
SHA, this cannot be discharged afterwards: there is no revision to compare
against, so `code_identical.jl` structurally cannot answer it. `worktree_dirty`
is therefore rendered beside the SHA rather than dropped, and an artifact that
does not record the flag says so — silence would be indistinguishable from a
verified-clean tree, which is the whole defect.

**`src_dirty` narrows that verdict, and is read when present.** `git_provenance`
records two flags from the same call: `worktree_dirty` over the whole tree, and
`src_dirty` from `git status --porcelain --untracked-files=no -- src`. Where the
second is `false`, no tracked file under `src/` differed, so the recorded SHA
*does* describe the package source that ran and the full-strength sentence above
would be false about the code the numbers are a measurement of. This function
therefore renders the narrower claim in that case, and the unqualified one only
when `src_dirty` is `true` or absent — the pre-`src_dirty` artifacts, where
nothing narrows it.

Read a `worktree_dirty = true` beside a `src_dirty = false` with the trap
documented on `git_provenance` in `docs/benchmark/common.jl` in mind: a harness
that splats `git_provenance()` *inside* its own `open(path, "w")` block sees the
output file it is truncating and records `true` over a clean tree. That shape
dirties only the results path, never `src/`, so it produces exactly this pair.

**This caption is the later of two checks, not the only one.** It reads a
committed artifact, so it can only fire once the bad flag has been recorded —
and it cannot fire on a harness's *first* run at all, since that write creates
an untracked file and `--untracked-files=no` ignores it by design. The earlier
half is `docs/benchmark/provenance_ordering.jl`, which parses every harness
under `docs/benchmark/` and `bench/` and goes red on the commit that introduces
the shape. Do not try to extend this function to cover that: it never sees the
harness source, only what the harness wrote.

It is rendered rather than refused because a dirty measurement is not worthless,
only unattributable, and some are kept deliberately as history: the `b5c7dee`
boxed-spec base carries `worktree_dirty = true` and is `SUPERSEDED` on purpose.
Erroring would take the build red on artifacts nobody intends to re-run. Zero
rows is the opposite case — no information at all — which is why
[`md_table`](@ref) refuses that one instead.

**The SHA this renders is provenance, not currency.** It says which revision was
measured, which stays true forever; it says nothing about whether that revision
still describes the sampler. Those come apart silently and in the reassuring
direction — the caption keeps naming a real commit, the table keeps rendering,
and the build stays green while every number on the page describes code that no
longer exists.

Do not try to close that here. `makedocs` has no history to consult, and on CI
the checkout is depth 1, so an artifact's base SHA does not even resolve — a
currency check wired into the docs build would be red for the wrong reason on
every run. The question is answered out of band, by
`docs/benchmark/artifact_currency.jl`, which walks every tracked artifact and
asks `code_identical.jl` whether `src/` at the recorded SHA still defines the
same methods as `src/` at a given revision. Run it after anything lands in
`src/`; nothing on this side can notice for you.
"""
function provenance(d::AbstractDict; harness::AbstractString)
    cfg = get(d, "config", nothing)
    field(k) = cfg isa AbstractDict ? get(cfg, k, get(d, k, nothing)) : get(d, k, nothing)
    sha = field("warmuphmc_sha")
    where_ = sha === nothing ? "an **unrecorded** WarmupHMC revision" :
             "WarmupHMC `$(first(string(sha), 7))`"
    bits = ["Measured on $(where_)"]
    jl = field("julia")
    jl === nothing || push!(bits, "Julia $(jl)")
    bt = field("blas_threads")
    bt === nothing || push!(bits, "$(bt) BLAS thread$(bt == 1 ? "" : "s")")
    base = string(join(bits, ", "), ", by `", harness, "`.")
    dirty = field("worktree_dirty")
    dirty === true && return base * (field("src_dirty") === false ?
        " **Recorded from a worktree with uncommitted changes, though none under" *
        " `src/`** — so the revision named above does describe the package source" *
        " that ran. What differed was elsewhere in the tree, which this flag does" *
        " not localise further; the harness itself is one such place." :
        " **Recorded from a worktree with uncommitted changes**, so the revision" *
        " named above does not describe the code that ran — and no revision does.")
    sha === nothing || dirty !== nothing ||
        return base * " (Worktree cleanliness was not recorded.)"
    base
end

"""
    num(x; sig=4) -> String

Format a number for a table cell. Magnitudes at or above 1000 are rounded to a
whole number rather than shown in scientific notation — these are mostly
nanosecond counts, where `448636` is readable and `4.486e+05` is not.
"""
function num(x; sig::Int = 4)
    x === nothing && return "—"
    x isa Integer && return string(x)
    v = Float64(x)
    isfinite(v) || return "—"
    abs(v) >= 1000 && return string(Int(round(v)))
    v == round(v) && return string(Int(round(v)))
    Printf.@sprintf("%.*g", sig, v)
end

"""
    load_harness(relpath)

Make a benchmark script's derivation functions callable from an `@eval` block,
by including `docs/benchmark/<relpath>` into the calling module.

This exists so that a *derived* figure — a paired summary, a conclusion — is
computed from the rows at docs-build time by the same code the benchmark uses,
instead of being stored alongside them where it can drift. Rows are the data;
everything else is a function of the rows.

Two constraints on a file loaded this way, because the docs environment is not
the benchmark environment:

  * **It must only define things.** Anything that runs at include time — reading
    `ARGS`, writing output — runs during the docs build. Keep the driver and the
    derivation in separate files, as `summarize.jl` already does.
  * **It may only depend on what `docs/Project.toml` has**: `JSON`, `Markdown`,
    `Printf`, `Statistics`. Pulling in BridgeStan or PosteriorDB to render a
    table would make the docs build depend on the whole measurement stack.
"""
function load_harness(relpath::AbstractString)
    path = normpath(joinpath(@__DIR__, "benchmark", relpath))
    isfile(path) || error("""
        Benchmark harness not found: $(path)

        An `@eval` block asked for a derivation script under docs/benchmark/.
        Either it was not committed, or a page names it wrongly.
        """)
    Base.include(Base.@__MODULE__, path)
end

"""
    vega_figure(spec; caption="") -> Markdown.MD

Render a Vega-Lite `spec` (anything `JSON.json` accepts) as a figure, by
emitting it as a fenced ` ```vega-lite ` block. `setupVegaFigures` in
`docs/src/.vitepress/theme/vega-figure.ts` finds the rendered block and swaps a
chart in; the runtimes are CDN tags in `config.mts`.

**A code fence is not a stylistic choice — it is the only construct that
survives this pipeline.** The obvious shape, an `@eval` block returning a
`<div data-spec="…">`, cannot work, and both layers that break it break it
silently:

  * `Markdown.parse` destroys the JSON before anything sees it. `\$schema` is
    read as inline math and `ns_const` as emphasis, so the text that reaches
    the AST is already corrupt — measured, not feared.
  * DocumenterVitepress escapes `<` and `>` in *every* text node
    (`escape_markdown_text`, `writer.jl`), by design, because Vue would
    otherwise parse a bare `<` as a tag. A div therefore arrives at the browser
    as `&lt;div&gt;`.

Raw HTML reaches the page only through `@raw html`, which is a static fence and
cannot carry a computed value. Fenced code is the one path a generated string
crosses untouched: markdown does not interpret inside it, and VitePress marks
code blocks `v-pre`, so Vue does not either.

Figures are DERIVED, never checked in. Build the spec from the rows at
docs-build time — see [`load_harness`](@ref) — for the same reason a summary
table is derived: a spec stored beside the rows it plots is a second copy of
them that nothing forces to agree, and it goes stale in the silent direction,
because a chart still renders when its numbers are old.

If the runtime is unreachable the block stays a readable JSON dump rather than
becoming a blank gap, which is the right failure: the figure's data is still
on the page.
"""
function vega_figure(spec; caption::AbstractString = "")
    parts = Any[Markdown.Code("vega-lite", JSON.json(spec))]
    isempty(caption) || append!(parts, Markdown.parse("*" * caption * "*").content)
    Markdown.MD(parts)
end

"""
    md_table(headers, rows) -> Markdown.MD

Build a markdown table. Cells are passed through `string`, so pre-format
anything that needs it (see [`num`](@ref)).

**Zero rows is an error, not an empty table.** A headers-only table is valid
markdown, so it renders as a blank table and the build exits 0 — an artifact
that still parses but has lost its rows would ship as an empty table with
nothing anywhere saying so. [`load_results`](@ref) only guards the file being
*absent*; this guards it being present and empty.

Measured before this check existed: emptying `rows` in one artifact and
rebuilding took the build down only because a *derivation* called `median` on
an empty array two sections further down. The three tables above it rendered
blank and reported nothing. The catch was incidental to what the harness
happened to compute, which is not a guarantee.

If an empty result is meaningful somewhere, branch on `isempty` in the `@eval`
block and emit prose saying so — the derived blocks in `adaptive-centering.md`
do exactly that, and prose is the honest rendering of "there is nothing here".
"""
function md_table(headers::AbstractVector, rows::AbstractVector)
    isempty(rows) && error("""
        md_table was given zero rows (headers: $(join(string.(headers), ", "))).

        This is refused rather than rendered, because a headers-only table is
        valid markdown: the page would show a blank table and the build would
        succeed. If the underlying artifact can legitimately be empty, branch on
        `isempty` in the @eval block and emit prose instead of a table.
        """)
    io = IOBuffer()
    println(io, "| ", join(string.(headers), " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for r in rows
        length(r) == length(headers) ||
            error("row has $(length(r)) cells, expected $(length(headers)): $(r)")
        println(io, "| ", join(string.(r), " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end
