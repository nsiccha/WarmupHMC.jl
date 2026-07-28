module WarmupHMCWeb

using HTMXObjects
using DynamicObjects
using Treebars
import PosteriorDB
using WarmupHMC
import AdvancedHMC
using LogDensityProblems
using MCMCDiagnosticTools
using Statistics
using BridgeStan, StanLogDensityProblems, JSON
using DifferentiationInterface
using Random
using LinearAlgebra
using TestModules

if @isdefined(IS_TESTING)
    include("test/runtests.jl")
end

include("posteriordb_reparametrizations.jl")


# ============================================================
# Benchmark evidence — one JSON, one renderer, two presentations
# ============================================================
#
# The benchmark drivers under `docs/benchmark/` write rows-only JSON into
# `docs/benchmark/results/`. That file is the SINGLE durable source. This
# app renders it live, and `@include record_gallery` (bottom of the file)
# freezes those renders into `docs/src/public/live-whmc/`, which the
# VitePress build serves. No table is ever retyped into prose, so the docs
# cannot drift from the run that produced them.
#
# Adding evidence is therefore a one-file act: drop a JSON in, and it
# appears in the app and in the docs with no change here. Everything below
# is deliberately schema-agnostic for that reason — see `benchmark_load`.

benchmark_results_dir() =
    joinpath(dirname(dirname(@__DIR__)), "docs", "benchmark", "results")

"""
    benchmark_key(relpath) -> String
    benchmark_path(key)    -> String

Round-trip a results file between its path under [`benchmark_results_dir`](@ref)
and its route segment. Results live one directory deep (`after/runs.json`), so
the separator is flattened to `~` — unreserved in RFC 3986, and absent from
every driver-emitted name (which use `-` and `_`).
"""
benchmark_key(relpath) = replace(replace(relpath, ".json" => ""), '/' => '~')
benchmark_path(key) =
    joinpath(benchmark_results_dir(), replace(String(key), '~' => '/') * ".json")

"""
    benchmark_keys() -> Vector{String}

Every results JSON currently on disk, sorted. Recomputed on each call rather
than cached: the drivers write here while the app is running, and a stale
listing would silently omit fresh evidence. Returns empty if the directory is
absent (the package is usable without a docs checkout).
"""
benchmark_keys() = begin
    root = benchmark_results_dir()
    isdir(root) || return String[]
    keys = String[]
    for (dir, _, files) in walkdir(root), file in files
        endswith(file, ".json") || continue
        push!(keys, benchmark_key(relpath(joinpath(dir, file), root)))
    end
    sort!(keys)
end

# Provenance keys the drivers stamp, in the order worth reading them. Anything
# else scalar is appended alphabetically, so a new field shows up unprompted
# rather than being silently dropped.
const BENCHMARK_PROVENANCE_ORDER =
    ["warmuphmc_sha", "julia", "blas_threads", "n_calls", "n_seeds", "rounds",
     "n_draws_floor", "note"]

# Row columns worth seeing first; the rest follow alphabetically. Column order
# must be deterministic — these renders get frozen into checked-in static HTML,
# and JSON objects parse into unordered `Dict`s.
const BENCHMARK_COLUMN_ORDER = ["target", "dim", "arm", "seed", "backend"]

_bench_order(names, priority) = vcat(
    [n for n in priority if n in names],
    sort([n for n in names if !(n in priority)]),
)

_bench_cell(::Nothing) = ""
_bench_cell(x::Bool) = string(x)
_bench_cell(x::Integer) = string(x)
# Round for reading, then drop a trailing `.0` — `184.0` and `445001.0` are
# counts and nanoseconds, not measurements to 1 decimal. Rounding happens
# BEFORE the integrality test so `445000.9805` lands on `445001`, not
# `445001.0`. Full precision stays in the JSON, which is the source.
_bench_cell(x::AbstractFloat) = begin
    isfinite(x) || return string(x)
    r = abs(x) >= 100 ? round(x; digits=1) : round(x; sigdigits=4)
    r == round(r) && abs(r) < 1e15 ? string(Int(round(r))) : string(r)
end
_bench_cell(x::AbstractVector) = join(_bench_cell.(x), ", ")
_bench_cell(x::AbstractDict) =
    join(["$k=$(_bench_cell(x[k]))" for k in sort(collect(keys(x)))], " ")
_bench_cell(x) = string(x)

"""
    benchmark_table(rows) -> NamedTuple of vectors

Turn a JSON array of row objects into a Tables.jl-compatible table for
[`render_table`](@ref). Columns are the union over all rows — a driver that
adds a field mid-series still renders, with `""` where the field is absent —
ordered by `BENCHMARK_COLUMN_ORDER` then alphabetically.
"""
benchmark_table(rows) = begin
    names = String[]
    for row in rows, k in keys(row)
        k in names || push!(names, k)
    end
    cols = _bench_order(names, BENCHMARK_COLUMN_ORDER)
    NamedTuple(Symbol(c) => [_bench_cell(get(row, c, nothing)) for row in rows]
               for c in cols)
end

"""
    benchmark_load(key) -> (; provenance, tables)

Read one results JSON and split it into scalar provenance and named row tables.

Both shapes the drivers emit are handled without naming either: a bare array of
rows (`after/gradient_overhead.json`), and an object carrying scalar provenance
alongside one or more arrays of rows (`rows`, `runs`, `captures`). The rule is
structural — an array-of-objects field is a table, every other field is
provenance — so a new driver needs no change here. That is the point: this
renderer must not become a second place where a result's schema is written down.
"""
benchmark_load(key) = begin
    parsed = JSON.parsefile(benchmark_path(key))
    if parsed isa AbstractVector
        return (; provenance = Pair{String,String}[],
                  tables = ["rows" => benchmark_table(parsed)])
    end
    tables = Pair{String,Any}[]
    scalars = String[]
    for (k, v) in parsed
        if v isa AbstractVector && !isempty(v) && all(x -> x isa AbstractDict, v)
            push!(tables, k => benchmark_table(v))
        else
            push!(scalars, k)
        end
    end
    sort!(tables; by=first)
    (; provenance = [k => _bench_cell(parsed[k])
                     for k in _bench_order(scalars, BENCHMARK_PROVENANCE_ORDER)],
       tables = tables)
end


# --- Web app ---

# ============================================================
# AppData — the global singleton holding all data + per-(model, method)
# computations. Long-running ops (Stan compile, MCMC sampling) live here
# so their per-key disk-cached results survive across per-request
# AppContext instances. `cache_type=:parallel` so the in-memory IP cache
# also dedupes concurrent requests for the same key.
# ============================================================

@dynamicstruct struct WhmcAppData

    __cache_path__ = joinpath(dirname(dirname(@__DIR__)), "web", "cache")

    pdb = PosteriorDB.database()

    @cached posterior_names = sort([
        Symbol(name) for name in PosteriorDB.posterior_names(pdb)
        if !isnothing(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, name)), "stan"))
    ])

    # === Recording (mirrors AlgebraOfVegaGallery §2b canonical form) ===
    # Holding these on the cached AppData (rather than inline in the
    # `@include record_gallery`) keeps the canonical AoV/htmxo-gallery
    # shape and lets the path list be reused / inspected from outside the
    # AppContext (e.g. from `record!` directly, or from a docs script).
    recording_dir   = joinpath(dirname(dirname(@__DIR__)), "docs", "src", "public", "live-whmc")
    recording_base  = get(ENV, "RECORD_BASE_PREFIX", "/WarmupHMC.jl/dev/live-whmc")
    recording_paths = vcat(
        ["/", "/gallery"],
        ["/posteriors/$name" for name in posterior_names],
        ["/benchmarks"],
        # Enumerated from disk, not listed by hand: a benchmark driver that
        # writes a new results JSON gets recorded into the docs without anyone
        # editing this file. See the benchmark-evidence block at the top.
        ["/benchmark/$key" for key in benchmark_keys()],
    )

    @struct posterior(name::Symbol) = begin
        seed     = 42
        n_draws  = 100

        # `name` is a Symbol so it survives `cache_segment`'s `maybehash` as a
        # readable directory name. PosteriorDB still wants a String; convert
        # once here and reuse the underlying handle for everything below.
        pdb_posterior = PosteriorDB.posterior(pdb, String(name))
        problem = StanLogDensityProblems.StanProblem(
            PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(pdb_posterior), "stan")),
            PosteriorDB.load(PosteriorDB.dataset(pdb_posterior), String);
            nan_on_error=true, make_args=["STAN_THREADS=TRUE"], warn=false)
        dimension = LogDensityProblems.dimension(problem)

        # === Per-method DOs ===

        @struct compile = begin
            label = "Compiles"
            @cached value = begin
                t0 = time()
                LogDensityProblems.dimension(problem)
                (elapsed = time() - t0,)
            end
            (; elapsed) = value
        end

        @struct sample = begin
            label = "WarmupHMC"
            rng   = Xoshiro(seed)
            @cached v"1" value = WarmupHMC.count_and_time(problem) do cp
                adaptive_warmup_mcmc(rng, cp; n_draws)
            end
            (; elapsed, n_evaluations) = value
            # `result` is renamed on the way in: `posterior` already has an indexed
            # `result(method)` property, and an inline `@struct` child sees its parent's
            # scope, so a bare `result` here would redefine it.
            sampler_result = value.result
            draws       = sampler_result.posterior_position
            n_divergent = sampler_result.n_divergent_samples
        end

        @struct dynamichmc = begin
            label = "DynamicHMC"
            rng   = Xoshiro(seed)
            @cached v"1" value = WarmupHMC.count_and_time(problem) do cp
                WarmupHMC.DynamicHMC.mcmc_with_warmup(rng, cp, n_draws;
                    reporter = WarmupHMC.DynamicHMC.NoProgressReport())
            end
            (; elapsed, n_evaluations) = value
            # `result` is renamed on the way in: `posterior` already has an indexed
            # `result(method)` property, and an inline `@struct` child sees its parent's
            # scope, so a bare `result` here would redefine it.
            sampler_result = value.result
            draws       = sampler_result.posterior_matrix
            n_divergent = count(s -> WarmupHMC.DynamicHMC.is_divergent(s.termination),
                                sampler_result.tree_statistics)
        end

        @struct advancedhmc = begin
            label    = "AdvancedHMC"
            rng      = Xoshiro(seed)
            # Match DynamicHMC's default warmup budget for a fair comparison
            # against the other samplers.
            n_adapts = 1000
            @cached v"1" value = WarmupHMC.count_and_time(problem) do cp
                metric      = AdvancedHMC.DiagEuclideanMetric(Float64, dimension)
                hamiltonian = AdvancedHMC.Hamiltonian(metric, cp)
                integrator  = AdvancedHMC.Leapfrog(0.1)
                kernel      = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
                                  integrator, AdvancedHMC.GeneralisedNoUTurn(10, 1000.0)))
                adaptor     = AdvancedHMC.StanHMCAdaptor(
                                  AdvancedHMC.MassMatrixAdaptor(metric),
                                  AdvancedHMC.StepSizeAdaptor(0.8, integrator))
                theta_init  = randn(rng, dimension)
                AdvancedHMC.sample(rng, hamiltonian, kernel, theta_init,
                    n_draws + n_adapts, adaptor, n_adapts;
                    drop_warmup=true, verbose=false, progress=false)
            end
            (; elapsed, n_evaluations) = value
            # `result` is renamed on the way in: `posterior` already has an indexed
            # `result(method)` property, and an inline `@struct` child sees its parent's
            # scope, so a bare `result` here would redefine it.
            sampler_result = value.result
            θs, stats   = sampler_result
            draws       = reduce(hcat, θs)
            n_divergent = sum(s.numerical_error for s in stats)
        end

        @struct reparam = begin
            label = "Reparam"
            rng   = Xoshiro(seed)

            # Delegate to the richer per-posterior reparametrization table in
            # posteriordb_reparametrizations.jl. The function takes a String
            # posterior name, the dimension, and the loaded Stan JSON data.
            # It always returns an IndexedReparametrization (empty for unknown
            # posteriors, non-empty when a spec exists).
            spec = reparametrization(String(name), dimension,
                                     PosteriorDB.load(PosteriorDB.dataset(pdb_posterior)))
            no_spec = isempty(spec.pairs)

            @cached v"1" value = begin
                # Harness choice, NOT a recommendation: ForwardDiff is already
                # here transitively via Pathfinder, and no reverse-mode backend
                # is loadable in this env. The documented backend for a
                # ReparametrizedProblem is reverse mode, and needs BOTH
                # `mode=set_runtime_activity(Reverse)` and
                # `function_annotation=Const`; a bare `AutoEnzyme()` throws on
                # the first gradient — see `docs/src/reparametrization.md`.
                rp   = ReparametrizedProblem(spec, problem, AutoForwardDiff())
                init = WarmupHMC.initialize_mcmc(problem, missing; rng, progress=nothing)
                WarmupHMC.count_and_time(rp) do cp
                    adaptive_warmup_mcmc(rng, cp; n_draws, init)
                end
            end
            (; elapsed, n_evaluations) = value
            # `result` is renamed on the way in: `posterior` already has an indexed
            # `result(method)` property, and an inline `@struct` child sees its parent's
            # scope, so a bare `result` here would redefine it.
            sampler_result = value.result
            draws       = sampler_result.posterior_position
            n_divergent = sampler_result.n_divergent_samples
            centering   = [(idx, v.source.c) for (idx, v) in spec.pairs]

            # Reparam-specific composed card: result html + centering line,
            # or a "Run" button when unstarted, or "" when no spec exists.
            section = if no_spec
                ""
            elseif (@cache_status value) == :unstarted
                h.section(
                    h.p(h.strong("Reparam: "),
                        h.a("Run"; hx_post="/posteriors/$name/result/reparam/run",
                            hx_target="closest div", hx_swap="outerHTML")))
            else
                centering_info = !isempty(centering) ?
                    h.p(h.strong("Final centering: "),
                        join(["[$idx] = $(round(c; digits=3))" for (idx, c) in centering[1:min(8,end)]], ", "),
                        length(centering) > 8 ? ", ..." : "") : ""
                h.section(__parent__.result(:reparam).html, centering_info)
            end
        end

        detail_id = "detail-$name"
        toggle    = "on click toggle [@hidden] on #$detail_id"

        # === Rendering ===

        @struct result(method::Symbol) = begin
            m      = getproperty(__parent__, method)
            status = @cache_status m.value
            label  = m.label
            run_url = "/posteriors/$name/result/$method/run"

            ess_vals      = MCMCDiagnosticTools.ess(reshape(m.draws', (:, 1, dimension)))
            median_ess    = median(ess_vals)
            min_ess       = minimum(ess_vals)
            elapsed       = m.elapsed
            n_divergent   = m.n_divergent
            n_evaluations = m.n_evaluations

            # Encodes status into the cell text directly so the per-sampler
            # status column can be dropped: number on success, "FAIL" on
            # failure, "-" when unstarted.
            formatted(name::Symbol; digits=1, suffix="") =
                status == :ready   ? "$(round(getproperty(__self__, name); digits))$suffix" :
                status == :started ? "FAIL" : "-"

            # Cells in the body row. Cells whose `data-status` is anything
            # other than `success` carry a click→run handler; PASS cells just
            # toggle the detail row. The batch-runner (search Enter handler)
            # picks them up via `td[data-status]:not([data-status="success"])`,
            # so no separate marker class is needed.
            run_handler = "on htmx:afterOnLoad if me.hasAttribute('data-batch') is false then remove [@hidden] from #$detail_id end remove [@data-batch] from me"

            # Compile-status cell (PASS/FAIL/-) for the "Compiles" column.
            status_cell() =
                status == :ready ?
                    h.td("PASS"; data_status="success", _=toggle) :
                status == :started ?
                    h.td("FAIL"; data_status="error",
                         hx_post=run_url, hx_target="#$detail_id", hx_swap="innerHTML",
                         _=run_handler) :
                    h.td("-"; data_status="muted",
                         hx_post=run_url, hx_target="#$detail_id", hx_swap="innerHTML",
                         _=run_handler)

            # Per-sampler metric cell. On `:ready` the cell toggles the detail
            # row; otherwise clicking posts the run URL.
            metric_cell(text) =
                status == :ready ?
                    h.td(text; _=toggle) :
                status == :started ?
                    h.td(text; data_status="error",
                         hx_post=run_url, hx_target="#$detail_id", hx_swap="innerHTML",
                         _=run_handler) :
                    h.td(text; data_status="muted",
                         hx_post=run_url, hx_target="#$detail_id", hx_swap="innerHTML",
                         _=run_handler)

            # The 3 metric cells (min ESS, # grad, time) for a sampler row.
            metric_cells() = (
                metric_cell(formatted(:min_ess)),
                metric_cell(formatted(:n_evaluations; digits=0)),
                metric_cell(formatted(:elapsed; digits=2, suffix="s")),
            )

            html = if status == :unstarted
                ""
            elseif status == :started
                h.section(
                    h.p(h.strong(label, ": "), status_badge(:failed; label="FAIL")),
                    h.p(h.em("Click the row's FAIL cell or re-run via the check route to view the error.")),
                )
            elseif method == :compile
                h.section(
                    h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
                    h.p(h.strong("Dimension: "), dimension),
                )
            else
                h.section(
                    h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
                    h.p(h.strong("Dimension: "), dimension),
                    h.p(h.strong("Draws: "), size(m.draws, 2),
                        " | ", h.strong("Min ESS: "),    formatted(:min_ess),
                        " | ", h.strong("Median ESS: "), formatted(:median_ess),
                        " | ", h.strong("Time: "),       formatted(:elapsed; digits=2, suffix="s"),
                        " | ", h.strong("Divergent: "),  n_divergent),
                )
            end

            # Mutating actions on this method's cache.
            force!() = begin
                status == :started && @clear_cache! m.value
                m.value
            end

            clear!() = begin
                @clear_cache! m.value
                "Cleared $method cache for $name"
            end
        end

        # Per-row cells used by `summary_row` (and re-rendered into the
        # OOB-swap response shape `summary_row => "row-$name"` by routes).
        # Layout: Posterior | Compiles | (ESS, Time) × 3 samplers = 8 cols.
        # ESS/Time cells double as the per-sampler trigger: when status is
        # `:ready` the cell toggles the detail row; otherwise it `hx_post`s
        # the corresponding `/posteriors/<name>/result/<m>/run` to (re-)run.
        row_cells = [
            h.td(name; _=toggle),
            result(:compile).status_cell(),
            result(:sample).metric_cells()...,
            result(:dynamichmc).metric_cells()...,
            result(:advancedhmc).metric_cells()...,
        ]

        # The bare row; the table wraps it together with a hidden sibling
        # for the expanded detail card. Routes that need to OOB-swap this
        # row return `summary_row => "row-$name"` (HTMX.jl's Pair handling
        # auto-adds hx_swap_oob and templates around table elements).
        summary_row = h.tr(row_cells...; id="row-$name")

        detail_content = begin
            r_compile     = result(:compile)
            r_sample      = result(:sample)
            r_dynamichmc  = result(:dynamichmc)
            r_advancedhmc = result(:advancedhmc)
            statuses = (r_compile.status, r_sample.status, r_dynamichmc.status, r_advancedhmc.status)
            any_fail = any(==(:started), statuses)
            all_pass_or_unstarted = all(s -> s in (:ready, :unstarted), statuses)
            banner_status = any_fail ? "error" :
                            all_pass_or_unstarted ? "success" :
                            "neutral"
            h.td(; colspan="11")(
                h.div(; class="htmxo-status-banner", data_status=banner_status)(
                    h.h4(name),
                    r_compile.html,
                    r_sample.html,
                    reparam.section,
                    r_dynamichmc.html,
                    r_advancedhmc.html,
                )
            )
        end

        # True iff any per-method `value` cache file exists for this posterior.
        # Used by `overview` to float touched posteriors to the top of the table.
        any_cached = @is_cached(compile.value) || @is_cached(sample.value) ||
                     @is_cached(dynamichmc.value) || @is_cached(advancedhmc.value)

        # Compact card for the `/gallery` view: title + per-method status
        # pills + deep link to `/posteriors/$name`. Cheap to render — only reads
        # `@cache_status m.value` per method, never triggers compute.
        gallery_card = let methods = (:compile, :sample, :dynamichmc, :advancedhmc, :reparam)
            h.article(
                h.h4(h.a(name; href="/posteriors/$name")),
                h.ul(
                    [let r = result(method); s = r.status
                        h.li(
                            h.span(r.label, ": "),
                            status_badge(s == :ready  ? :done   :
                                         s == :started ? :failed : :info;
                                label = s == :ready  ? "PASS" :
                                        s == :started ? "FAIL" : "-"))
                     end for method in methods]...,
                ),
                h.p(h.a("View detail →"; href="/posteriors/$name")),
            )
        end
    end
end

const APPDATA = WhmcAppData()

# ============================================================
# AppContext — ephemeral per-request DO. Holds rendering / page chrome
# and routes; reads from `__appdata__` for everything data-side.
# ============================================================

@htmx struct AppContext
    __appdata__ = APPDATA

    # NOTE: deliberately NOT destructuring `(; posterior, posterior_names) =
    # __appdata__` — that would bind the names as AppContext properties and
    # collide with any future `@get posterior(...)` / similar route. Reach
    # into `__appdata__.…` explicitly at the call site instead. (See the
    # corresponding note in StanBlocks/PosteriorDBWeb.jl.)

    # === Page (sidebar, page chrome) ===

    __page__(content) = htmx(
        app_layout(
            nav_sidebar(["Table" => "/", "Gallery" => "/gallery"]; prefix=string(__self__)),
            content,
        ),
        h.style(".htmxo-status-banner section { margin-bottom: 0.5rem; }");
        pico_version="2",
    )

    # ============================================================
    # === Routes ===
    # ============================================================

    @get index() = h.div(
        h.h2("WarmupHMC PosteriorDB Dashboard ($(length(__appdata__.posterior_names)) posteriors)"),
        h.p(
            # Examples link disabled — example routes commented out during refactor.
            # hx_link("/examples")("Examples (progress demo)"),
            # " | ",
            h.a(href=__self__/"tests")("Tests"),
        ),
        h.input(;
            type="search",
            id="search",
            placeholder="Filter posteriors...",
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove [@hidden] from row else add [@hidden] to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not([hidden])' set target to null for cell in <td[data-status]:not([data-status='success'])/> in row if target is null set target to cell end end if target is not null set @data-batch to '' on target send click to target end end end",
        ),
        h.table(class="htmxo-sortable-table striped"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)", rowspan="2"),
                    h.th("Compiles";  _="on click call sortTable(1, me)", rowspan="2"),
                    h.th("WarmupHMC";   colspan="3"),
                    h.th("DynamicHMC";  colspan="3"),
                    h.th("AdvancedHMC"; colspan="3"),
                ),
                h.tr(
                    h.th("Min ESS"; _="on click call sortTable(2, me)"),
                    h.th("# Grad";  _="on click call sortTable(3, me)"),
                    h.th("Time";    _="on click call sortTable(4, me)"),
                    h.th("Min ESS"; _="on click call sortTable(5, me)"),
                    h.th("# Grad";  _="on click call sortTable(6, me)"),
                    h.th("Time";    _="on click call sortTable(7, me)"),
                    h.th("Min ESS"; _="on click call sortTable(8, me)"),
                    h.th("# Grad";  _="on click call sortTable(9, me)"),
                    h.th("Time";    _="on click call sortTable(10, me)"),
                ),
            ),
            h.tbody(reduce(vcat, [begin
                                      p = __appdata__.posterior(name)
                                      [p.summary_row, h.tr(; id="detail-$name", hidden="")(p.detail_content)]
                                  end
                                  for name in sort(__appdata__.posterior_names;
                                                  by=name -> !__appdata__.posterior(name).any_cached)];
                            init=[])...; id="posterior-tbody")
        ),
        sortable_table_js(),
        sortable_table_styles(),
    )

    # Card-grid view. Same posteriors as `/`, just laid out as compact
    # cards with status pills + deep-links instead of a sortable table.
    @get gallery() = h.div(
        h.h2("WarmupHMC Posterior Gallery ($(length(__appdata__.posterior_names)) posteriors)"),
        h.div(; class="htmxo-gallery")(
            [__appdata__.posterior(name).gallery_card
             for name in __appdata__.posterior_names]...,
        ),
    )

    # GET `/benchmarks` — index of the evidence under `docs/benchmark/results/`.
    # Enumerated from disk per request (not cached), so a driver run mid-session
    # shows up without restarting the app.
    @get benchmarks() = begin
        ks = benchmark_keys()
        summaries = map(ks) do key
            loaded = benchmark_load(key)
            prov = Dict(loaded.provenance)
            (; file    = key,
               sha     = first(get(prov, "warmuphmc_sha", ""), 7),
               tables  = join([name for (name, _) in loaded.tables], ", "),
               rows    = sum(t -> length(first(t)), (t for (_, t) in loaded.tables); init=0),
               note    = get(prov, "note", ""))
        end
        h.div(
            htmxo_breadcrumb([("Table", "/", "/"), ("Benchmarks", nothing, nothing)]),
            h.h2("Benchmark evidence ($(length(ks)) result files)"),
            h.p("""Each row is one JSON written by a driver under docs/benchmark/. \
                 That file is the source: this page renders it, and the docs build \
                 serves a static recording of this same render. Nothing is retyped, \
                 so a table in the documentation cannot drift from the run that \
                 produced it."""),
            h.p(h.strong("Check the SHA before citing a row."), """ A results file is \
                 a snapshot of the tree it was measured on, and this directory keeps \
                 superseded runs alongside current ones on purpose — the comparison \
                 is often the point."""),
            isempty(summaries) ?
                h.p(h.em("No results checked in under docs/benchmark/results/.")) :
                render_table(summaries; id="benchmark-index", download=false,
                    cell=(v, c, _) -> c === :file ?
                        h.a(string(v); href=__self__/"benchmark/$v") : string(v)),
            sortable_table_js(), sortable_table_styles(), download_table_js(),
        )
    end

    # Per-file view: /benchmark/<key>, where <key> is the results path with
    # `/` flattened to `~` (see `benchmark_key`).
    #
    # DO NOT collapse `json_path` into its single use below. An INDEXED
    # `@include` whose body holds exactly one statement is unwrapped by
    # `_unwrap_short_form_body` (HTMXObjects.jl:773) and reinterpreted as the
    # short-form EXTERNAL mount `@include benchmark(key) = SomeChild(key)` —
    # the parser wraps a short-form method body in a `:block` too, so the head
    # alone cannot tell the two apart and the statement count is the only
    # signal. The lone `@get` is then spliced in as a call expression, and
    # precompilation dies with `UndefVarError: @get not defined`. Two
    # statements make it an inline sub-router, which is what this is. The
    # sibling `@include posteriors` / `@include results` escape it only by
    # happening to have more than one statement. Reported as a snag to
    # HTMXObjects.
    @include benchmark(key::Symbol) = begin
        json_path = benchmark_path(key)

        @get index() = if !isfile(json_path)
            h.div(
                htmxo_breadcrumb([("Table", string(__parent__), nothing),
                                  ("Benchmarks", string(__parent__/"benchmarks"), nothing)]),
                h.p(h.em("No results file `$(key)`.")),
            )
        else
            loaded = benchmark_load(key)
            h.div(
                htmxo_breadcrumb([("Table", string(__parent__), nothing),
                                  ("Benchmarks", string(__parent__/"benchmarks"), nothing),
                                  (String(key), nothing, nothing)]),
                h.h2(String(key)),
                isempty(loaded.provenance) ? "" :
                    h.dl([[h.dt(name), h.dd(value)]
                          for (name, value) in loaded.provenance]...),
                [[h.h3(name), render_table(table; id="benchmark-$key-$name",
                                           download_filename="$key-$name.csv")]
                 for (name, table) in loaded.tables]...,
                sortable_table_js(), sortable_table_styles(), download_table_js(),
            )
        end
    end

    # Drop all in-memory caches on the singleton appdata. Useful after
    # property/value-shape edits that Revise tracks at the method level but
    # can't invalidate per cached instance.
    @delete cache() = (clear_mem_caches!(__appdata__); "ok")

    # Per-posterior view: routes mounted under /posteriors/<name>/…
    @include posteriors(name::Symbol) = begin
        @get index() = h.div(
            htmxo_breadcrumb([
                ("Table", "/", "/"),
                (name, nothing, nothing),
            ]),
            __appdata__.posterior(name).detail_content,
        )

        # Per-(name, method) actions: /posteriors/<name>/result/<method>/{run,cache}
        @include result(method::Symbol) = begin
            # POST /posteriors/<name>/result/<method>/run — (re)compute that result.
            # `:reparam` returns its own section; the four samplers return
            # the shared (detail_content, OOB-row-swap) tuple.
            @post run() = begin
                p = __appdata__.posterior(name)
                p.result(method).force!()
                method == :reparam ? p.reparam.section :
                    [p.detail_content, p.summary_row => "row-$name"]
            end
            @delete cache() = __appdata__.posterior(name).result(method).clear!()
        end
    end

    # Per-method view: cross-posterior aggregates mounted under /results/<method>/…
    @include results(method::Symbol) = begin
        # Posteriors filtered by their `result(method).status`. Stays a fresh
        # call (no @memo! / brackets) so disk-status changes are visible.
        matching(status_filter) = begin
            names = Symbol[]
            for name in __appdata__.posterior_names
                s = __appdata__.posterior(name).result(method).status
                ok = if     status_filter == "pass";      s == :ready
                     elseif status_filter == "fail";      s == :started
                     elseif status_filter == "unchecked"; s == :unstarted
                     else;                                true
                     end
                ok && push!(names, name)
            end
            names
        end

        @get  posteriors(status="all") = join(matching(status), "\n")
        @post run(status="fail") = begin
            for name in matching(status)
                __appdata__.posterior(name).result(method).force!()
            end
            "ok"
        end
    end

    @include structure = HTMXObjects.StructureRoutes(; root=AppContext)

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)

    # GET `/record_gallery` — drives `RECORDING_STATE.record` to dump
    # `/` (overview) + `/posteriors/$name` for every posterior into
    # `docs/src/public/live-whmc/` as static HTML (full + HX shapes). The
    # docs build picks them up from there. Paths / dir / base live on
    # `WhmcAppData` to match the canonical AoV form (htmxo-gallery §2b).
    @include record_gallery = RecordingRoutes(;
        app_type    = AppContext,
        paths       = __appdata__.recording_paths,
        record_dir  = __appdata__.recording_dir,
        record_base = __appdata__.recording_base,
        label       = "Recording WHMC dashboard",
    )
end

function __init__()
    route!(AppContext())
end

end # module WarmupHMCWeb
