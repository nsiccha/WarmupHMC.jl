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

static_dir() = joinpath(dirname(@__DIR__), "static")

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


function status_cell(status::Symbol)
    status == :ready ? h.td("PASS"; class="u-text-success u-text-bold") :
    status == :started ? h.td("FAIL"; class="u-text-error u-text-bold") :
    h.td("-"; class="u-text-muted")
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

#= --- Example computations with progress ---
DISABLED during AppContext refactor: depends on the now-inlined module-level
`stan_problem` helper. Restore by reintroducing `stan_problem(pn)` (or by
having these helpers reach into AppContext for a per-posterior problem).

function run_parallel_pathfinder(posterior_name; n_chains=4, seed=42, progress=nothing)
    problem = stan_problem(posterior_name)
    dim = LogDensityProblems.dimension(problem)
    rngs = [Xoshiro(seed + i) for i in 1:n_chains]
    # Each chain's initialize_mcmc creates a "Pathfinder" child with per-iteration updates
    with_progress(progress, n_chains; description="Pathfinder x$n_chains") do p
        results = Vector{Any}(undef, n_chains)
        Threads.@threads for i in 1:n_chains
            # Wrap each chain in its own progress subtree
            results[i] = with_progress(p; description="Chain $i") do cp
                WarmupHMC.initialize_mcmc(problem, missing; rng=rngs[i], progress=cp)
            end
            update_progress!(p)
        end
        (n_chains=n_chains, dimension=dim, results=results)
    end
end

function run_parallel_sampling(posterior_name; n_chains=4, n_draws=100, seed=42, progress=nothing)
    problem = stan_problem(posterior_name)
    rngs = [Xoshiro(seed + i) for i in 1:n_chains]
    # adaptive_warmup_mcmc multi-chain already creates per-chain progress:
    # "MCMC" (N=n_chains) → "MCMC.1" (N=n_draws+50, with divergences/stepsize/ESS) etc.
    results = adaptive_warmup_mcmc(rngs, problem; n_draws, progress, monitor_ess=true)
    n_divergent = sum(r -> r.n_divergent_samples, results)
    n_samples = sum(r -> size(r.posterior_position, 2), results)
    dims = size(first(results).posterior_position, 1)
    (n_chains=n_chains, dimension=dims, n_draws=n_samples,
     n_divergent=n_divergent, results=results)
end

_sample_r_path() = joinpath(homedir(), "github", "nsiccha", "CmdstanR.jl", "sample.R")

function _count_csv_draws(path)
    !isfile(path) && return 0
    n = 0
    open(path) do io
        for line in eachline(io)
            startswith(line, '#') && continue
            n += 1
        end
    end
    max(0, n - 1)  # subtract header row
end

function run_cmdstan_sampling(posterior_name; n_chains=4, iter_warmup=1000, iter_sampling=1000, seed=42, progress=nothing)
    posterior = PosteriorDB.posterior(pdb, posterior_name)
    stan_path = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan"))
    data_json = PosteriorDB.load(PosteriorDB.dataset(posterior), String)

    output_dir = mktempdir()
    output_basename = "cmdstan_$(posterior_name)"
    data_path = joinpath(output_dir, "data.json")
    config_path = joinpath(output_dir, "config.json")

    open(data_path, "w") do fd; print(fd, data_json); end
    config = Dict(
        "seed" => seed,
        "output_dir" => output_dir,
        "output_basename" => output_basename,
        "chains" => n_chains,
        "parallel_chains" => n_chains,
        "iter_warmup" => iter_warmup,
        "iter_sampling" => iter_sampling,
        "save_warmup" => true,
        "sig_figs" => 18,
    )
    open(config_path, "w") do fd; JSON.print(fd, config); end

    total_per_chain = iter_warmup + iter_sampling
    script_path = _sample_r_path()

    with_progress(progress, n_chains; description="CmdStan x$n_chains") do p
        # Create per-chain progress nodes
        chain_progress = [initialize_progress!(p, total_per_chain; description="Chain $i") for i in 1:n_chains]

        # Launch Rscript in background
        proc = run(`Rscript $script_path $stan_path $data_path $config_path`; wait=false)

        # Monitor CSV files until process exits
        while process_running(proc)
            for i in 1:n_chains
                csv_path = joinpath(output_dir, "$(output_basename)-$(i).csv")
                draws = _count_csv_draws(csv_path)
                update_progress!(chain_progress[i], draws)
            end
            sleep(0.5)
        end

        # Final update
        for i in 1:n_chains
            csv_path = joinpath(output_dir, "$(output_basename)-$(i).csv")
            draws = _count_csv_draws(csv_path)
            update_progress!(chain_progress[i], draws)
            finalize_progress!(chain_progress[i])
        end

        # Check exit status
        success(proc) || error("CmdStan sampling failed (exit code $(proc.exitcode))")

        # Collect results
        csv_paths = filter(f -> startswith(basename(f), output_basename) && endswith(f, ".csv"), readdir(output_dir; join=true))
        sort!(csv_paths)
        total_draws = sum(csv_paths) do f; _count_csv_draws(f); end
        (n_chains=n_chains, n_draws=total_draws, iter_warmup=iter_warmup,
         iter_sampling=iter_sampling, output_dir=output_dir, csv_paths=csv_paths)
    end
end

@dynamicstruct struct ExampleComputations
    __status__ = initialize_progress!(:state; description="Examples")

    pathfinder(pn) = run_parallel_pathfinder(pn; progress=__status__)
    sampling(pn) = run_parallel_sampling(pn; progress=__status__)
    cmdstan(pn) = run_cmdstan_sampling(pn; progress=__status__)
end
_examples = ExampleComputations(; cache_type=:parallel)


function _posterior_select(names, id)
    h.select(; id, name="pn")(
        [h.option(pn; value=pn) for pn in names]...
    )
end
=#


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

        # Reparametrization spec dispatch (inlined from the old free
        # `posterior_reparametrization`). The richer mapping in
        # web/src/posteriordb_reparametrizations.jl can replace this later.
        reparam_spec = if startswith(pn, "funnel")
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
            @cached value = begin
                isnothing(reparam_spec) && error("No reparametrization defined for $pn")
                rp   = ReparametrizedProblem(reparam_spec, problem, AutoForwardDiff())
                init = WarmupHMC.initialize_mcmc(problem, missing; rng, progress=nothing)
                t0   = time()
                raw  = adaptive_warmup_mcmc(rng, rp; n_draws, init)
                (elapsed     = time() - t0,
                 draws       = raw.posterior_position,
                 n_divergent = raw.n_divergent_samples,
                 centering   = [(idx, v.source.c) for (idx, v) in reparam_spec.pairs])
            end
            (; elapsed, draws, n_divergent, centering) = value
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
        end

        # Reparam card composes result html + centering line + special "Run" button when unstarted
        reparam_section = if isnothing(reparam_spec)
            ""
        elseif (@cache_status reparam.value) == :unstarted
            h.div(; class="u-mb-2")(
                h.p(h.strong("Reparam: "),
                    h.a("Run"; hx_get=__appdata__/"check/reparam/$pn",
                        hx_target="closest div", hx_swap="outerHTML",
                        class="u-pointer")))
        else
            centering = reparam.value.centering
            centering_info = !isempty(centering) ?
                h.p(h.strong("Final centering: "),
                    join(["[$idx] = $(round(c; digits=3))" for (idx, c) in centering[1:min(8,end)]], ", "),
                    length(centering) > 8 ? ", ..." : "") : ""
            h.div(; class="u-mb-2")(result(:reparam).html, centering_info)
        end

        # Per-row cells, computed once and reused by `summary_row` and `summary_row_swapped`.
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

        summary_row = [h.tr(row_cells...; id="row-$pn"),
                       h.tr(; id="detail-$pn", class="hidden")]

        summary_row_swapped = h.tr(row_cells...; id="row-$pn", hx_swap_oob="outerHTML:#row-$pn")

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
                    reparam_section,
                    r_dynamichmc.html,
                    r_advancedhmc.html,
                )
            )
        end

        # === Mutating actions ===

        # Force (re)compute a single check, clearing prior-failure cache first.
        # On failure compute_property re-throws → route safety wrapper renders the error.
        force(method::Symbol) = begin
            m = getproperty(__self__, method)
            (@cache_status m.value) == :started && @clear_cache! m.value
            m.value
        end

        clear(method::Symbol) = begin
            m = getproperty(__self__, method)
            @clear_cache! m.value
            "Cleared $method cache for $pn"
        end
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
            h.tbody(reduce(vcat, [(@memo posterior(pn)).summary_row for pn in sort(posterior_names; by=pn -> begin
                p = @memo posterior(pn)
                !(@is_cached(p.compile.value) || @is_cached(p.sample.value) || @is_cached(p.dynamichmc.value) || @is_cached(p.advancedhmc.value))
            end)]; init=[])...; id="posterior-tbody")
        ),
        sortable_table_js(),
        h.style(".hidden { display: none; } tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    sidebar_html = h.aside(; id="sidebar")(
        h.nav(
            h.div(h.strong("WarmupHMC"); class="sidebar-title"),
            h.ul(
                h.li(h.a("Table"; class="nav-item", data_nav="table",
                    hx_get=__self__, hx_target="#content", hx_swap="innerHTML",
                    hx_push_url="/",
                    _="on click remove .nav-active from .nav-item then add .nav-active to me")),
            ),
        ),
    )

    __page__(content) = htmx(
        h.div(; class="app-layout")(
            sidebar_html,
            h.div(; class="app-main")(h.div(content; id="content"))
        ),
        h.style("""
            /* Override Pico defaults for dashboard-appropriate sizing */
            :root {
                --pico-font-size: 87.5%;
                --pico-spacing: 0.75rem;
                --pico-border-radius: 0.25rem;
            }
            body { max-width: none; padding: 0; margin: 0; }
            body > .app-layout { max-width: none; }

            .app-layout {
                display: flex;
                min-height: 100vh;
            }

            #sidebar {
                width: 160px;
                min-width: 160px;
                padding: 12px 8px;
                border-right: 1px solid var(--pico-muted-border-color);
                background: var(--pico-card-background-color);
            }
            .sidebar-title {
                padding: 8px 12px;
                margin-bottom: 12px;
                font-size: 1em;
            }

            .app-main {
                flex: 1;
                padding: 20px 24px;
                overflow-x: auto;
                min-width: 0;
            }

            /* Sidebar nav items — override Pico's <a> styling */
            .nav-item {
                display: block;
                padding: 6px 12px;
                margin-bottom: 2px;
                border-radius: 4px;
                font-size: 0.9em;
                cursor: pointer;
                text-decoration: none !important;
                color: var(--pico-color) !important;
            }
            .nav-item:hover {
                background: var(--pico-table-row-stripped-background-color);
                color: var(--pico-color) !important;
            }
            .nav-active {
                font-weight: bold !important;
                background: var(--pico-primary-background) !important;
                color: var(--pico-primary-inverse) !important;
            }

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

        """),
        h.script(raw"""
            function updateNav() {
                var path = window.location.pathname;
                document.querySelectorAll('.nav-item').forEach(function(el) { el.classList.remove('nav-active'); });
                var el = document.querySelector('[data-nav="table"]');
                if (el) el.classList.add('nav-active');
            }
            document.addEventListener('DOMContentLoaded', updateNav);
            document.body.addEventListener('htmx:pushedIntoHistory', updateNav);
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

    @get serve_static(filename) = begin
        filepath = joinpath(static_dir(), filename)
        isfile(filepath) || return HTTP.Response(404, body="Not found")
        ext = lowercase(splitext(filename)[2])
        ct = ext == ".js" ? "application/javascript" : ext == ".css" ? "text/css" : "text/plain"
        HTTP.Response(200, ["Content-Type" => ct], body=read(filepath))
    end

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
        p.force(m)
        m == :reparam ? p.reparam_section :
            [p.detail_content, h.template(p.summary_row_swapped)]
    end

    @get clear(check, pn) = (@memo posterior(pn)).clear(Symbol(check))

    @get filter(check, status="all") = join(filtered_names[check, status], "\n")

    @get recheck(check, status="fail") = begin
        targets = filtered_names[check, status]
        results = String[]
        for pn in targets
            (@memo posterior(pn)).force(Symbol(check))
            push!(results, "PASS $pn")
        end
        join(results, "\n")
    end

    #= --- Examples (DISABLED during AppContext refactor) ---
    # The example_* routes depend on the now-disabled run_parallel_pathfinder /
    # run_parallel_sampling / run_cmdstan_sampling / ExampleComputations / _examples
    # at module scope. Restore those (or rebuild them on top of `posterior(pn).problem`)
    # to re-enable.
    examples_content = h.div(
        h.h2("Examples: WarmupHMC + Treebars Progress"),
        h.p("Demonstrates parallel computation with live progress reporting."),
        h.div(
            _posterior_select(posterior_names, "examples-select"),
            h.fieldset(; role="group")(
                h.button("Pathfinder"; hx_get=__self__/"example_pathfinder", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
                h.button("WarmupHMC"; hx_get=__self__/"example_sampling", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
                h.button("CmdStan"; hx_get=__self__/"example_cmdstan", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
            ),
        ),
        h.div(; id="examples-result"),
    )

    @get examples = h.div(; id="content")(
        breadcrumb([("Table", "/fragment_table", "/"), ("Examples", nothing, nothing)]),
        examples_content,
    )

    @get example_pathfinder(; pn="", force::Bool=false) = begin
        isempty(pn) && return h.p("Select a posterior above.")
        fetchindex(_examples.pathfinder, pn; force) do rv, status
            if rv isa Task
                h.div(; hx_get=query_url("/example_pathfinder"; pn), hx_trigger="every 200ms", hx_swap="outerHTML")(
                    h.article(h.header("Pathfinder ($pn) — running..."), htmx_render_children(status))
                )
            else
                h.article(
                    h.header("Pathfinder ($pn) — done"),
                    h.p("$(rv.n_chains) chains, dimension $(rv.dimension)"),
                    h.p("All chains initialized successfully."),
                    h.button("Rerun"; hx_get=query_url("/example_pathfinder"; pn, force=true), hx_target="closest article", hx_swap="outerHTML"),
                )
            end
        end
    end

    @get example_sampling(; pn="", force::Bool=false) = begin
        isempty(pn) && return h.p("Select a posterior above.")
        fetchindex(_examples.sampling, pn; force) do rv, status
            if rv isa Task
                h.div(; hx_get=query_url("/example_sampling"; pn), hx_trigger="every 200ms", hx_swap="outerHTML")(
                    h.article(h.header("Sampling ($pn) — running..."), htmx_render_children(status))
                )
            else
                h.article(
                    h.header("Sampling ($pn) — done"),
                    h.p("$(rv.n_chains) chains, dimension $(rv.dimension), $(rv.n_draws) total draws"),
                    h.p("Divergent: $(rv.n_divergent)"),
                    h.button("Rerun"; hx_get=query_url("/example_sampling"; pn, force=true), hx_target="closest article", hx_swap="outerHTML"),
                )
            end
        end
    end

    @get example_cmdstan(; pn="", force::Bool=false) = begin
        isempty(pn) && return h.p("Select a posterior above.")
        fetchindex(_examples.cmdstan, pn; force) do rv, status
            if rv isa Task
                h.div(; hx_get=query_url("/example_cmdstan"; pn), hx_trigger="every 500ms", hx_swap="outerHTML")(
                    h.article(h.header("CmdStan ($pn) — running..."), htmx_render_children(status))
                )
            else
                h.article(
                    h.header("CmdStan ($pn) — done"),
                    h.p("$(rv.n_chains) chains, $(rv.n_draws) total draws ($(rv.iter_warmup) warmup + $(rv.iter_sampling) sampling per chain)"),
                    h.button("Rerun"; hx_get=query_url("/example_cmdstan"; pn, force=true), hx_target="closest article", hx_swap="outerHTML"),
                )
            end
        end
    end
    =# # --- end Examples (DISABLED) ---

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
end

function __init__()
    route!(AppContext())
end

end # module WarmupHMCWeb
