# The evidence appendix: every checked-in results file, rendered from itself.
#
# `tables.jl` renders SELECTED figures into the prose pages — a named file, named
# columns, a sentence built around them. This renders ALL of them, naming none:
# it walks `docs/benchmark/results/`, and the shape of each file decides how it
# is displayed. A driver that lands a new results JSON gets an appendix section
# with no edit here and no edit to `evidence.md`.
#
# The rule is structural, and deliberately the only rule:
#
#   * a field holding an array of objects is a TABLE (`rows`, `runs`, `arms`,
#     `captures`, `overheads`, …), columns being the union over its rows;
#   * every other field is PROVENANCE;
#   * a bare top-level array is a single table named `rows`.
#
# Nothing here knows what a column means, which is the point. The moment this
# file names a driver's fields it becomes a second place that driver's schema is
# written down, and the two drift.
#
# THE ONE DUPLICATION THIS REPO ACCEPTS, AND WHY
# ----------------------------------------------
# `web/src/WarmupHMCWeb.jl` applies these same rules to serve `/benchmarks` and
# `/benchmark/<key>`, returning HTMXObjects semantic values that carry a Markdown
# projection. Calling that projection from here would collapse the two into one
# implementation — and it was built and working (16kB page, all 13 files) before
# this file replaced it.
#
# It is not used, because HTMXObjects.jl is a PRIVATE repo. Everything the docs
# build currently clones — `TestModules.jl`, `Treebars.jl` — is public, so the
# `PAT_TOKEN` step in `Docs.yml` has never actually been exercised against a
# private repo. Making the deployment depend on it, to render an appendix, risks
# the whole docs build on an unverified secret. `docs/Project.toml` stays at
# JSON/Markdown/Printf/Statistics, and this runs inside Documenter's `@eval` with
# no extra CI step at all.
#
# So the DATA has one source; the PRESENTATION has two renderers. That is a real
# cost — a formatting change wants making twice — and the compensating property
# is that neither renderer can invent a number: both read the same JSON, and
# `make.jl` fails the build if a file is missing or malformed.
#
# WHAT DELETION DOES, AND WHY THERE IS NO MANIFEST
# ------------------------------------------------
# A file this page merely LISTS can be deleted with a green build — its section
# just stops appearing. `WarmupHMC:reparam-docs` established that by experiment
# (moved `prep_cost.json` aside, rebuilt, `makedocs` succeeded and the rendered
# page contained no `prep_cost`), against a sentence on `evidence.md` that
# claimed the opposite. The sentence was the fix; the gap is deliberate.
#
# The alternative was a checked-in manifest of expected filenames, diffed
# against `results_dir()`. It was declined for the reason stated at the top of
# this file: a manifest is a SECOND place the set of results files is written
# down, and it would drift from the directory exactly as a hard-coded column
# list drifts from a driver's schema. It would also fight a documented property
# of this directory — superseded runs are kept beside current ones on purpose,
# so the set is meant to change.
#
# The gate that does exist is citation: `tables.jl` NAMES the file it reads, so
# deleting anything a prose page quotes fails the build. Uncited files are, by
# construction, evidence no page depends on.

Base.include(Base.@__MODULE__, joinpath(@__DIR__, "tables.jl"))

"""
    evidence_keys() -> Vector{String}

Every results file under `docs/benchmark/results/`, as a key: the path relative
to that directory, without `.json`, with `/` flattened to `~`. The flattening
matches the web app's route parameter, so `after/runs.json` is `after~runs` here
and at `/benchmark/after~runs`.
"""
function evidence_keys()
    root = results_dir()
    isdir(root) || error("no $(root) — the results directory is not checked in")
    keys = String[]
    for (dir, _, files) in walkdir(root), file in files
        endswith(file, ".json") || continue
        rel = relpath(joinpath(dir, file), root)
        push!(keys, replace(replace(rel, ".json" => ""), Base.Filesystem.path_separator => "~"))
    end
    isempty(keys) && error("""
        No results files under $(root).

        The evidence appendix is generated from them, and a page that renders as
        an empty appendix would deploy green with its tables silently gone.
        """)
    sort!(keys)
end

evidence_path(key) = joinpath(results_dir(), replace(String(key), '~' => '/') * ".json")

"""
    evidence_is_series(v) -> Bool

Whether an object-valued field is a set of measurement SERIES rather than
metadata: every one of its values is a non-empty numeric array.

The obvious discriminator — "all its values are scalars" — is wrong here, and
checking rather than assuming is what showed it. `capture_boxing.json`'s
`timings_ns` holds four named arrays of raw A/B timings, and rendering those as
provenance presented measurement data as metadata. But `nonlinear_weighting`'s
`config` ALSO holds non-scalars (`seeds` is an array, `targets` and
`resolved_sampler_defaults` are objects), so a scalars-only rule would have
thrown the genuine metadata block out of provenance to fix the cosmetic case.

Requiring EVERY value to be a numeric array separates them cleanly: `config` is
a mix of strings, arrays and objects and stays provenance; `timings_ns` is
uniformly numeric arrays and becomes a table. Still a rule about shape, not
about either name.
"""
_evidence_numeric_vector(x) =
    x isa AbstractVector && !isempty(x) && all(y -> y isa Real && !(y isa Bool), x)
evidence_is_series(v::AbstractDict) =
    !isempty(v) && all(_evidence_numeric_vector, values(v))
evidence_is_series(_) = false

"""
    evidence_split(d) -> (provenance, tables)

Split a parsed results file into scalar provenance and named row tables.

An OBJECT-valued field is flattened into provenance rather than shown as one
mashed `k=v k=v` cell — unless `evidence_is_series` says it is measurement data,
in which case it becomes a table. That is how a harness carrying its metadata
under `config` — the shape `provenance` in `tables.jl` now prefers, since one
copy of `warmuphmc_sha` cannot disagree with a duplicate of itself — renders the
same as the older flat shape. The nested entry wins on a name collision,
matching `tables.jl`. Stated as a rule about objects rather than about `config`,
so it holds for whatever the next harness calls its grouping.
"""
function evidence_split(d)
    d isa AbstractVector && return (Pair{String,Any}[], ["rows" => d])
    tables = Pair{String,Any}[]
    scalars = Pair{String,Any}[]
    nested = Pair{String,Any}[]
    for k in sort(collect(keys(d)))
        v = d[k]
        if v isa AbstractVector && !isempty(v) && all(x -> x isa AbstractDict, v)
            push!(tables, k => v)
        elseif evidence_is_series(v)
            # One row per series. Two columns and no invented third: the array
            # index is almost certainly a round number here, but this file does
            # not know that, and a column header asserting it would be this
            # renderer claiming to understand a driver's schema.
            push!(tables, k => [Dict("series" => nk, "values" => v[nk])
                                for nk in sort(collect(keys(v)))])
        elseif v isa AbstractDict
            append!(nested, [string(nk) => v[nk] for nk in sort(collect(keys(v)))])
        else
            push!(scalars, k => v)
        end
    end
    nested_names = Set(first.(nested))
    (vcat(nested, [p for p in scalars if !(first(p) in nested_names)]), tables)
end

# Provenance fields worth reading first; anything else follows alphabetically.
const EVIDENCE_PROVENANCE_ORDER =
    ["warmuphmc_sha", "julia", "host", "blas_threads", "n_seeds", "n_calls",
     "rounds", "n_draws_floor", "note"]
const EVIDENCE_COLUMN_ORDER =
    ["series", "target", "dim", "arm", "seed", "backend", "source"]

_evidence_order(names, priority) = vcat(
    [n for n in priority if n in names],
    sort([n for n in names if !(n in priority)]),
)

# `num` handles the numeric cases; these are the shapes a results file also
# carries. A vector cell is joined rather than dropped — `con_names` and friends
# are short label lists, and an empty one must read as empty, not as missing.
evidence_cell(::Nothing) = "—"
evidence_cell(x::Bool) = x ? "yes" : "no"
# Values are escaped for the same reason names are, and this half was found the
# hard way SECOND: escaping only the keys left `nonlinear_weighting.jl`,
# `all_good_leaves`, `max_tree_depth` and `ad_backend_check` rendering as
# emphasis inside provenance text. Values carry far more identifiers than keys do
# — filenames, symbol names, whole reproduction recipes — so this is the larger
# exposure of the two. `|` on top of the inline escapes, since a cell sits inside
# a pipe table.
evidence_cell(x::AbstractString) =
    isempty(x) ? "—" : replace(esc_md(x), "|" => "\\|")
evidence_cell(x::Real) = num(x)
evidence_cell(x::AbstractVector) = isempty(x) ? "—" : join(evidence_cell.(x), ", ")
evidence_cell(x::AbstractDict) =
    isempty(x) ? "—" :
    join(["$(esc_md(k))=$(evidence_cell(x[k]))" for k in sort(collect(keys(x)))], " ")
evidence_cell(x) = evidence_cell(string(x))

"Column names of a row table: the union over all rows, priority-ordered."
function evidence_columns(rows)
    names = String[]
    for row in rows, k in keys(row)
        k in names || push!(names, k)
    end
    _evidence_order(names, EVIDENCE_COLUMN_ORDER)
end

_md(s::AbstractString) = Markdown.parse(s)

"""
    esc_md(s) -> String

Escape markdown inline syntax in text that is INTERPOLATED into a string this
file then parses — file keys and column names, which come from filenames and
JSON keys and so are not under this file's control.

Two underscores in one word are emphasis, so `n_draws_floor` rendered as
n*draws*floor. This was **not** hypothetical and was not caught by looking at
file keys — those happen to carry one underscore each. Column and provenance
names do not, and eight of them shipped mangled on the page:
`n_draws_floor`, `max_grad_diff`, `ab_max_grad_diff`, `bare_ns_median`,
`reuse_grad_diff`, `const_over_fd_randn`, `const_over_fd_typical`,
`first_window_gradient_budget`. It surfaced from diffing the rendered output
across the change, not from reading the code.

The escape is consumed by the parser, so the rendered text and the anchor
VitePress derives from it are unchanged — for every character in the class
BELOW, which is not the class this function started with. Measured, one
character at a time, through `Markdown.parse`:

  * `` ` `` `*` `_` `[` `]` `{` `}` and `\\` are consumed, as documented;
  * **`<` and `>` are NOT.** A `\\>` ships the backslash VISIBLY — a spec
    string reading `x -> x[1]/2` rendered as `x -\\> x[1]/2`. Backslash
    escaping is simply not available for those two here, so they are mapped
    to HTML entities instead. That is also the safer target: VitePress runs
    the page through Vue, which parses a bare `<Foo>` in markdown as a
    component, and an entity cannot be mistaken for one.

The entity form costs one thing worth knowing: `Markdown`'s own HTML writer
re-escapes the `&`, so a local `show(MIME"text/html"(), …)` preview displays
`&amp;lt;`. The shipped path is markdown-out to VitePress, where it is a `<`.
"""
esc_md(s) = replace(string(s),
    r"([\\`*_\[\]{}])" => s"\\\1",
    '<' => "&lt;",
    '>' => "&gt;",
)
_join_md(parts) = Markdown.MD(reduce(vcat, (p.content for p in parts); init = Any[]))

"""
    md_evidence(key; max_rows = 40) -> Markdown.MD

One results file as a section: its provenance, then each of its row tables.

A table longer than `max_rows` is DECLINED rather than truncated. The four
`runs` files are raw per-run sampler logs — `linear_restart.json` alone holds
800 rows of 28 columns, several of them vectors — and the first 40 rows of a raw
log is neither the data nor a summary of it. Rendering every table in full was
measured at 1.7MB of markdown against 16kB with the cap, and it buried the small
tables the appendix exists to show. The declined section names the file and the
route that serves the rows instead.
"""
function md_evidence(key::AbstractString; max_rows::Int = 40)
    d = load_results(replace(String(key), '~' => '/') * ".json")
    scalars, tables = evidence_split(d)
    parts = Any[_md("## $(esc_md(key))\n")]

    isempty(scalars) ||
        push!(parts, _md(join(["- **$(esc_md(k))**: $(evidence_cell(v))" for (k, v) in
                               [p for n in _evidence_order(first.(scalars),
                                                           EVIDENCE_PROVENANCE_ORDER)
                                for p in scalars if first(p) == n]], "\n") * "\n"))

    for (name, rows) in tables
        # A bold label, NOT a `###` heading, and not by preference.
        # DocumenterVitepress's writer hardcodes `"\n# "` for a bare
        # `MarkdownAST.Heading` and ignores its level (writer.jl:1155 in
        # v0.2), so every heading produced by an `@eval` block renders as an
        # `<h1>` no matter what level it was built at. Headings from the page
        # SOURCE avoid this by arriving wrapped in a `Documenter.AnchoredHeader`.
        #
        # The file keys above can live with that — they are top-level sections
        # of an appendix and their anchors resolve correctly. A sub-table cannot:
        # `linear_restart.json` holds `arms` and `runs`, and as `<h1>`s they sit
        # in the page outline as siblings of `linear_restart` itself, reading as
        # two more results files. A bold label says the same thing and cannot
        # lie about the structure.
        length(tables) == 1 || push!(parts, _md("**$(esc_md(name))**\n"))
        if length(rows) > max_rows
            push!(parts, _md("""
                !!! note "$(length(rows)) rows — not inlined"
                    `$(name)` holds $(length(rows)) rows of \
                    $(length(evidence_columns(rows))) columns. That is raw \
                    measurement output, not a table to read: it is checked in at \
                    `docs/benchmark/results/$(replace(String(key), '~' => '/')).json`, \
                    the web app serves it sorted at `/benchmark/$(key)`, and the \
                    derived views live in the pages that cite them.
                """))
        else
            cols = evidence_columns(rows)
            push!(parts, md_table(esc_md.(cols),
                [[evidence_cell(get(row, c, nothing)) for c in cols] for row in rows]))
        end
    end
    _join_md(parts)
end

"""
    md_evidence_index() -> Markdown.MD

One row per results file: what it holds, and the revision it was measured on.
"""
function md_evidence_index()
    rows = map(evidence_keys()) do key
        d = load_results(replace(key, '~' => '/') * ".json")
        scalars, tables = evidence_split(d)
        prov = Dict(scalars)
        sha = get(prov, "warmuphmc_sha", nothing)
        [string("[`", key, "`](#", lowercase(replace(key, r"[^A-Za-z0-9]+" => "-")), ")"),
         sha === nothing ? "—" : string("`", first(string(sha), 7), "`"),
         join(["$(n) ($(length(t)))" for (n, t) in tables], ", "),
         get(prov, "host", "—")]
    end
    md_table(["file", "measured at", "tables (rows)", "host"], rows)
end

"""
    md_evidence_all(; max_rows = 40) -> Markdown.MD

The whole appendix: the index followed by every file.
"""
md_evidence_all(; max_rows::Int = 40) = _join_md(vcat(
    [md_evidence_index()],
    [md_evidence(key; max_rows) for key in evidence_keys()],
))
