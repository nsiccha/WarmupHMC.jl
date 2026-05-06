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

include("test/runtests.jl")


pdb = PosteriorDB.database()

# --- Web app ---

function breadcrumb(items)
    parts = []
    for (i, (label, frag_url, push_url)) in enumerate(items)
        i > 1 && push!(parts, h.span(" / "; class="bc-sep"))
        if isnothing(frag_url)
            push!(parts, h.span(label; class="bc-current"))
        else
            push!(parts, h.a(label;
                hx_get=frag_url, hx_target="#content", hx_swap="innerHTML",
                hx_push_url=push_url, class="bc-link"))
        end
    end
    h.nav(parts...; class="breadcrumbs", aria_label="breadcrumb")
end


function status_cell_clickable(status::Symbol, check_url, detail_id)
    if status == :ready
        return h.td("PASS"; class="check-cell u-pointer u-text-success u-text-bold",
            _="on click toggle .hidden on #$detail_id")
    elseif status == :started
        return h.td("FAIL"; class="check-cell u-pointer u-text-error u-text-bold",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    else
        return h.td("-"; class="check-cell u-pointer u-text-muted",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    end
end

@htmx struct AppContext

    # ============================================================
    # === Data ===
    # ============================================================

    cache_path = joinpath(dirname(dirname(@__DIR__)), "web", "cache")

    @cached posterior_names = sort([
        pn for pn in PosteriorDB.posterior_names(pdb)
        if !isnothing(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, pn)), "stan"))
    ])

    @struct posterior(pn) = begin
        seed     = 42
        n_draws  = 100

        # Routes deliver `pn` as a SubString; PosteriorDB.posterior wants String.
        # Convert once and reuse the underlying PosteriorDB handle.
        pdb_posterior = PosteriorDB.posterior(pdb, String(pn))
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
            @cached value = begin
                t0  = time()
                raw = adaptive_warmup_mcmc(rng, problem; n_draws)
                (elapsed     = time() - t0,
                 draws       = raw.posterior_position,
                 n_divergent = raw.n_divergent_samples)
            end
            (; elapsed, draws, n_divergent) = value
        end

        @struct dynamichmc = begin
            label = "DynamicHMC"
            rng   = Xoshiro(seed)
            @cached value = begin
                t0  = time()
                raw = WarmupHMC.DynamicHMC.mcmc_with_warmup(rng, problem, n_draws;
                          reporter = WarmupHMC.DynamicHMC.NoProgressReport())
                (elapsed     = time() - t0,
                 draws       = raw.posterior_matrix,
                 n_divergent = count(s -> WarmupHMC.DynamicHMC.is_divergent(s.termination),
                                     raw.tree_statistics))
            end
            (; elapsed, draws, n_divergent) = value
        end

        @struct advancedhmc = begin
            label    = "AdvancedHMC"
            rng      = Xoshiro(seed)
            n_adapts = 100
            @cached value = begin
                metric      = AdvancedHMC.DiagEuclideanMetric(Float64, dimension)
                hamiltonian = AdvancedHMC.Hamiltonian(metric, problem)
                integrator  = AdvancedHMC.Leapfrog(0.1)
                kernel      = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
                                  integrator, AdvancedHMC.GeneralisedNoUTurn(10, 1000.0)))
                adaptor     = AdvancedHMC.StanHMCAdaptor(
                                  AdvancedHMC.MassMatrixAdaptor(metric),
                                  AdvancedHMC.StepSizeAdaptor(0.8, integrator))
                theta_init  = randn(rng, dimension)
                t0          = time()
                θs, stats   = AdvancedHMC.sample(rng, hamiltonian, kernel, theta_init,
                                  n_draws + n_adapts, adaptor, n_adapts;
                                  drop_warmup=true, verbose=false, progress=false)
                (elapsed     = time() - t0,
                 draws       = reduce(hcat, θs),
                 n_divergent = sum(s.numerical_error for s in stats))
            end
            (; elapsed, draws, n_divergent) = value
        end

        @struct reparam = begin
            label = "Reparam"
            rng   = Xoshiro(seed)

            # Reparametrization spec dispatch (inlined from the old free
            # `posterior_reparametrization`). The richer mapping in
            # web/src/posteriordb_reparametrizations.jl can replace this later.
            spec = if startswith(pn, "funnel")
                IndexedReparametrization(2:dimension .=> Ref(Reparametrization(
                    PartiallyCentered(1.), PartiallyCentered(1.), 0., x->x[1])))
            elseif !isnothing(match(r"-eight_schools_(non|)centered", pn))
                c = endswith(pn, "noncentered") ? 0. : 1.
                IndexedReparametrization(1:8 .=> Ref(Reparametrization(
                    PartiallyCentered(c), PartiallyCentered(c), x->x[9], x->x[10])))
            elseif !isnothing(match(r"-radon_partially_pooled_(non|)centered", pn))
                J = PosteriorDB.load(PosteriorDB.dataset(pdb_posterior))["J"]
                c = endswith(pn, "noncentered") ? 0. : 1.
                IndexedReparametrization(map(1:J) do i
                    i => Reparametrization(PartiallyCentered(c), PartiallyCentered(c),
                                            x->x[J+1], x->x[J+2])
                end)
            else
                nothing
            end

            @cached value = begin
                isnothing(spec) && error("No reparametrization defined for $pn")
                rp   = ReparametrizedProblem(spec, problem, AutoForwardDiff())
                init = WarmupHMC.initialize_mcmc(problem, missing; rng, progress=nothing)
                t0   = time()
                raw  = adaptive_warmup_mcmc(rng, rp; n_draws, init)
                (elapsed     = time() - t0,
                 draws       = raw.posterior_position,
                 n_divergent = raw.n_divergent_samples,
                 centering   = [(idx, v.source.c) for (idx, v) in spec.pairs])
            end
            (; elapsed, draws, n_divergent, centering) = value

            # Reparam-specific composed card: result html + centering line,
            # or a "Run" button when unstarted, or "" when no spec exists.
            section = if isnothing(spec)
                ""
            elseif (@cache_status value) == :unstarted
                h.div(; class="u-mb-2")(
                    h.p(h.strong("Reparam: "),
                        h.a("Run"; hx_get=__appdata__/"check/reparam/$pn",
                            hx_target="closest div", hx_swap="outerHTML",
                            class="u-pointer")))
            else
                centering_info = !isempty(centering) ?
                    h.p(h.strong("Final centering: "),
                        join(["[$idx] = $(round(c; digits=3))" for (idx, c) in centering[1:min(8,end)]], ", "),
                        length(centering) > 8 ? ", ..." : "") : ""
                h.div(; class="u-mb-2")(__parent__.result(:reparam).html, centering_info)
            end
        end

        # === Rendering ===

        @struct result(method::Symbol) = begin
            m      = getproperty(__parent__, method)
            status = @cache_status m.value
            label  = m.label

            ess_vals    = MCMCDiagnosticTools.ess(reshape(m.draws', (:, 1, dimension)))
            median_ess  = median(ess_vals)
            min_ess     = minimum(ess_vals)
            elapsed     = m.elapsed
            n_divergent = m.n_divergent

            formatted(name::Symbol; digits=1, suffix="") =
                status == :ready ? "$(round(getproperty(__self__, name); digits))$suffix" : "-"

            html = if status == :unstarted
                ""
            elseif status == :started
                h.div(; class="u-mb-2")(
                    h.p(h.strong(label, ": "), status_badge(:failed; label="FAIL")),
                    h.p(h.em("Click the row's FAIL cell or re-run via the check route to view the error.")),
                )
            elseif method == :compile
                h.div(; class="u-mb-2")(
                    h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
                    h.p(h.strong("Dimension: "), dimension),
                )
            else
                h.div(; class="u-mb-2")(
                    h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
                    h.p(h.strong("Dimension: "), dimension),
                    h.p(h.strong("Draws: "), size(m.draws, 2),
                        " | ", h.strong("Min ESS: "),    formatted(:min_ess),
                        " | ", h.strong("Median ESS: "), formatted(:median_ess),
                        " | ", h.strong("Time: "),       formatted(:elapsed; digits=2, suffix="s"),
                        " | ", h.strong("Divergent: "),  n_divergent),
                )
            end

            # Mutating actions on this method's cache. Callers use the fresh
            # form (no `@memo`) so each invocation actually runs the body.
            force!() = begin
                status == :started && @clear_cache! m.value
                m.value
            end

            clear!() = begin
                @clear_cache! m.value
                "Cleared $method cache for $pn"
            end
        end

        # Per-row cells used by `summary_row` (and re-rendered into the
        # OOB-swap response shape `summary_row => "row-$pn"` by routes).
        row_cells = begin
            detail_id = "detail-$pn"
            toggle = "on click toggle .hidden on #$detail_id"
            r_compile, r_sample, r_dynamichmc, r_advancedhmc =
                result(:compile), result(:sample), result(:dynamichmc), result(:advancedhmc)
            [
                h.td(pn; class="u-pointer", _=toggle),
                status_cell_clickable(r_compile.status,     "/check/compile/$pn",     detail_id),
                status_cell_clickable(r_sample.status,      "/check/sample/$pn",      detail_id),
                h.td(r_sample.formatted(:median_ess);                       class="u-pointer", _=toggle),
                h.td(r_sample.formatted(:elapsed; digits=2, suffix="s");    class="u-pointer", _=toggle),
                status_cell_clickable(r_dynamichmc.status,  "/check/dynamichmc/$pn",  detail_id),
                h.td(r_dynamichmc.formatted(:median_ess);                   class="u-pointer", _=toggle),
                h.td(r_dynamichmc.formatted(:elapsed; digits=2, suffix="s");class="u-pointer", _=toggle),
                status_cell_clickable(r_advancedhmc.status, "/check/advancedhmc/$pn", detail_id),
                h.td(r_advancedhmc.formatted(:median_ess);                  class="u-pointer", _=toggle),
                h.td(r_advancedhmc.formatted(:elapsed; digits=2, suffix="s");class="u-pointer", _=toggle),
            ]
        end

        # The bare row; the table wraps it together with a hidden sibling
        # for the expanded detail card. Routes that need to OOB-swap this
        # row return `summary_row => "row-$pn"` (HTMX.jl's Pair handling
        # auto-adds hx_swap_oob and templates around table elements).
        summary_row = h.tr(row_cells...; id="row-$pn")

        detail_content = begin
            r_compile     = result(:compile)
            r_sample      = result(:sample)
            r_dynamichmc  = result(:dynamichmc)
            r_advancedhmc = result(:advancedhmc)
            statuses = (r_compile.status, r_sample.status, r_dynamichmc.status, r_advancedhmc.status)
            any_fail = any(==(:started), statuses)
            all_pass_or_unstarted = all(s -> s in (:ready, :unstarted), statuses)
            status_class = any_fail ? "u-status-callout u-status-error" :
                           all_pass_or_unstarted ? "u-status-callout u-status-success" :
                           "u-status-callout"
            h.td(; colspan="11", class="whmc-detail-cell")(
                h.div(; class=status_class)(
                    h.h4(pn),
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

    end

    # ============================================================
    # === Page (overview, sidebar, page chrome) ===
    # ============================================================

    overview = h.div(
        h.h2("WarmupHMC PosteriorDB Dashboard ($(length(posterior_names)) posteriors)"),
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
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove .hidden from row else add .hidden to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not(.hidden)' set target to null for cell in <td.check-cell/> in row if target is null and cell.textContent.trim() is not 'PASS' set target to cell end end if target is not null add .batch to target send click to target end end end",
            class="u-w-full u-mb-4",
        ),
        h.table(class="striped"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)", class="u-pointer", rowspan="2"),
                    h.th("Compiles"; _="on click call sortTable(1, me)", class="u-pointer", rowspan="2"),
                    h.th("WarmupHMC"; colspan="3", class="whmc-th-merge"),
                    h.th("DynamicHMC"; colspan="3", class="whmc-th-merge"),
                    h.th("AdvancedHMC"; colspan="3", class="whmc-th-merge"),
                ),
                h.tr(
                    h.th("Status"; _="on click call sortTable(2, me)", class="u-pointer"),
                    h.th("ESS"; _="on click call sortTable(3, me)", class="u-pointer"),
                    h.th("Time"; _="on click call sortTable(4, me)", class="u-pointer"),
                    h.th("Status"; _="on click call sortTable(5, me)", class="u-pointer"),
                    h.th("ESS"; _="on click call sortTable(6, me)", class="u-pointer"),
                    h.th("Time"; _="on click call sortTable(7, me)", class="u-pointer"),
                    h.th("Status"; _="on click call sortTable(8, me)", class="u-pointer"),
                    h.th("ESS"; _="on click call sortTable(9, me)", class="u-pointer"),
                    h.th("Time"; _="on click call sortTable(10, me)", class="u-pointer"),
                ),
            ),
            h.tbody(reduce(vcat, [begin
                                      p = @memo posterior(pn)
                                      [p.summary_row, h.tr(; id="detail-$pn", class="hidden")]
                                  end
                                  for pn in sort(posterior_names;
                                                  by=pn -> !(@memo posterior(pn)).any_cached)];
                            init=[])...; id="posterior-tbody")
        ),
        sortable_table_js(),
        h.style(".hidden { display: none; } tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    __page__(content) = htmx(
        h.div(; class="grid")(
            nav_sidebar(["Table" => "/"]; prefix=string(__self__)),
            h.div(content; id="content"),
        ),
        h.style("""
            /* Breadcrumbs */
            .breadcrumbs {
                font-size: 0.85em;
                margin-bottom: 16px;
                padding: 6px 0;
                border-bottom: 1px solid var(--pico-muted-border-color);
            }
            .bc-sep { color: var(--pico-muted-color); margin: 0 6px; }
            .bc-current { color: var(--pico-color); font-weight: 500; }
            .bc-link {
                text-decoration: none !important;
                color: var(--pico-primary) !important;
                cursor: pointer;
            }
            .bc-link:hover { text-decoration: underline !important; }

            /* WHMC table & detail card */
            .whmc-th-merge { text-align: center; border-bottom: none; }
            .whmc-detail-cell { padding: 0; border: none; }
        """);
        pico_version="2",
    )

    # Force (re)compute a single check, clearing prior-failure cache first.
    # On failure compute_property re-throws → route safety wrapper renders the error.
    filtered_names(check, status_filter) = begin
        names = String[]
        for pn in posterior_names
            p = @memo posterior(pn)
            s = (@memo p.result(Symbol(check))).status
            show = if status_filter == "pass"; s == :ready
            elseif status_filter == "fail"; s == :started
            elseif status_filter == "unchecked"; s == :unstarted
            else; true
            end
            show && push!(names, pn)
        end
        names
    end

    # ============================================================
    # === Routes ===
    # ============================================================

    @get index = overview

    @get model(pn) = h.div(; id="content")(
        breadcrumb([
            ("Table", "/", "/"),
            (pn, nothing, nothing),
        ]),
        (@memo posterior(pn)).detail_content,
    )

    # Collapsed check routes — one parameterized handler instead of five.
    # `:reparam` returns its own section; the four samplers return the
    # shared (detail_content, swapped-row template) tuple.
    @get check(method, pn) = begin
        p = @memo posterior(pn)
        m = Symbol(method)
        p.result(m).force!()
        m == :reparam ? p.reparam.section :
            [p.detail_content, p.summary_row => "row-$pn"]
    end

    @get clear(check, pn) = (@memo posterior(pn)).result(Symbol(check)).clear!()

    @get filter(check, status="all") = join(filtered_names[check, status], "\n")

    @get recheck(check, status="fail") = begin
        targets = filtered_names[check, status]
        results = String[]
        for pn in targets
            (@memo posterior(pn)).result(Symbol(check)).force!()
            push!(results, "PASS $pn")
        end
        join(results, "\n")
    end


    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
end

function __init__()
    route!(AppContext())
end

end # module WarmupHMCWeb
