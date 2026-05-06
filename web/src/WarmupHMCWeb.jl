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
import Pkg
using TestModules

include("test/runtests.jl")


pdb = PosteriorDB.database()

static_dir() = joinpath(dirname(@__DIR__), "static")

function stan_problem(posterior_name)
    posterior = PosteriorDB.posterior(pdb, posterior_name)
    StanLogDensityProblems.StanProblem(
        PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan")),
        PosteriorDB.load(PosteriorDB.dataset(posterior), String);
        nan_on_error=true, make_args=["STAN_THREADS=TRUE"], warn=false
    )
end

function posterior_reparametrization(posterior_name)
    posterior = PosteriorDB.posterior(pdb, posterior_name)
    jdata = PosteriorDB.load(PosteriorDB.dataset(posterior))
    if startswith(posterior_name, "funnel")
        dim = LogDensityProblems.dimension(stan_problem(posterior_name))
        IndexedReparametrization(2:dim .=> Ref(Reparametrization(
            PartiallyCentered(1.), PartiallyCentered(1.), 0., x->x[1]
        )))
    elseif !isnothing(match(r"-eight_schools_(non|)centered", posterior_name))
        c = endswith(posterior_name, "noncentered") ? 0. : 1.
        IndexedReparametrization(1:8 .=> Ref(Reparametrization(
            PartiallyCentered(c), PartiallyCentered(c), x->x[9], x->x[10]
        )))
    elseif !isnothing(match(r"-radon_partially_pooled_(non|)centered", posterior_name))
        J = jdata["J"]
        c = endswith(posterior_name, "noncentered") ? 0. : 1.
        IndexedReparametrization(map(1:J) do i
            i => Reparametrization(PartiallyCentered(c), PartiallyCentered(c), x->x[J+1], x->x[J+2])
        end)
    else
        nothing
    end
end

function compile_check(posterior_name)
    problem = stan_problem(posterior_name)
    dim = LogDensityProblems.dimension(problem)
    (dimension=dim,)
end

function run_sample(posterior_name; n_draws=100, seed=42)
    problem = stan_problem(posterior_name)
    dim = LogDensityProblems.dimension(problem)
    rng = Xoshiro(seed)
    t0 = time()
    result = adaptive_warmup_mcmc(rng, problem; n_draws)
    elapsed = time() - t0
    draws = result.posterior_position
    n = size(draws, 2)
    dim = size(draws, 1)
    ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
    min_ess = minimum(ess_vals)
    median_ess = median(ess_vals)
    draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
    (n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
     n_divergent=result.n_divergent_samples, draws_2d=draws_2d)
end

function run_sample_reparam(posterior_name; n_draws=100, seed=42)
    reparam = posterior_reparametrization(posterior_name)
    isnothing(reparam) && error("No reparametrization defined for $posterior_name")
    problem = stan_problem(posterior_name)
    rp = ReparametrizedProblem(reparam, problem, AutoForwardDiff())
    dim = LogDensityProblems.dimension(rp)
    rng = Xoshiro(seed)
    # Initialize Pathfinder on the unwrapped problem (Pathfinder uses ForwardDiff
    # which doesn't work through the reparametrization transform),
    # then pass the init to the reparametrized sampler
    init = WarmupHMC.initialize_mcmc(problem, missing; rng, progress=nothing)
    t0 = time()
    result = adaptive_warmup_mcmc(rng, rp; n_draws, init)
    elapsed = time() - t0
    draws = result.posterior_position
    n = size(draws, 2)
    ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
    min_ess = minimum(ess_vals)
    median_ess = median(ess_vals)
    draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
    centering = [(idx, v.source.c) for (idx, v) in reparam.pairs]
    (n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
     n_divergent=result.n_divergent_samples, draws_2d=draws_2d, centering=centering)
end

function run_sample_dynamichmc(posterior_name; n_draws=100, seed=42)
    problem = stan_problem(posterior_name)
    dim = LogDensityProblems.dimension(problem)
    rng = Xoshiro(seed)
    t0 = time()
    result = WarmupHMC.DynamicHMC.mcmc_with_warmup(rng, problem, n_draws; reporter=WarmupHMC.DynamicHMC.NoProgressReport())
    elapsed = time() - t0
    draws = result.posterior_matrix
    n = size(draws, 2)
    ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
    min_ess = minimum(ess_vals)
    median_ess = median(ess_vals)
    n_divergent = count(s -> WarmupHMC.DynamicHMC.is_divergent(s.termination), result.tree_statistics)
    draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
    (n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
     n_divergent=n_divergent, draws_2d=draws_2d)
end

function run_sample_advancedhmc(posterior_name; n_draws=100, n_adapts=100, seed=42)
    problem = stan_problem(posterior_name)
    dim = LogDensityProblems.dimension(problem)
    rng = Xoshiro(seed)
    metric = AdvancedHMC.DiagEuclideanMetric(Float64, dim)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, problem)
    integrator = AdvancedHMC.Leapfrog(0.1)
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn(10, 1000.0)))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(0.8, integrator)
    )
    theta_init = randn(rng, dim)
    t0 = time()
    θs, stats = AdvancedHMC.sample(rng, hamiltonian, kernel, theta_init, n_draws + n_adapts, adaptor, n_adapts; drop_warmup=true, verbose=false, progress=false)
    elapsed = time() - t0
    draws = reduce(hcat, θs)
    n = size(draws, 2)
    ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
    min_ess = minimum(ess_vals)
    median_ess = median(ess_vals)
    n_divergent = sum(s.numerical_error for s in stats)
    draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
    (n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
     n_divergent=n_divergent, draws_2d=draws_2d)
end

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

# --- Example computations with progress ---

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


@htmx struct AppContext
    
    cache_path = joinpath(dirname(dirname(@__DIR__)), "web", "cache")

    @cached posterior_names = sort([
        pn for pn in PosteriorDB.posterior_names(pdb)
        if !isnothing(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, pn)), "stan"))
    ])

    @cached compile_result(pn) = compile_check(pn)

    @cached sample_result(pn) = run_sample(pn)

    @cached dynamichmc_result(pn) = run_sample_dynamichmc(pn)

    @cached advancedhmc_result(pn) = run_sample_advancedhmc(pn)

    @cached reparam_result(pn) = run_sample_reparam(pn)

    # Three-state status from disk cache:
    # :ready     — succeeded, value cached
    # :started   — attempted but failed (or in flight); accessing re-runs
    # :unstarted — never attempted
    compile_status(pn) = @cache_status compile_result[pn]
    sample_status(pn) = @cache_status sample_result[pn]
    dynamichmc_status(pn) = @cache_status dynamichmc_result[pn]
    advancedhmc_status(pn) = @cache_status advancedhmc_result[pn]
    reparam_status(pn) = @cache_status reparam_result[pn]

    overview_row(pn) = begin
        c_status = compile_status[pn]
        s_status = sample_status[pn]
        d_status = dynamichmc_status[pn]
        a_status = advancedhmc_status[pn]
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"

        w_ess = s_status == :ready ? string(round(sample_result[pn].median_ess; digits=1)) : "-"
        w_time = s_status == :ready ? string(round(sample_result[pn].time; digits=2), "s") : "-"
        d_ess = d_status == :ready ? string(round(dynamichmc_result[pn].median_ess; digits=1)) : "-"
        d_time = d_status == :ready ? string(round(dynamichmc_result[pn].time; digits=2), "s") : "-"
        a_ess = a_status == :ready ? string(round(advancedhmc_result[pn].median_ess; digits=1)) : "-"
        a_time = a_status == :ready ? string(round(advancedhmc_result[pn].time; digits=2), "s") : "-"

        [h.tr(
            h.td(pn; class="u-pointer", _=toggle),
            status_cell_clickable(c_status, "/check_compile/$pn", detail_id),
            status_cell_clickable(s_status, "/check_sample/$pn", detail_id),
            h.td(w_ess; class="u-pointer", _=toggle),
            h.td(w_time; class="u-pointer", _=toggle),
            status_cell_clickable(d_status, "/check_dynamichmc/$pn", detail_id),
            h.td(d_ess; class="u-pointer", _=toggle),
            h.td(d_time; class="u-pointer", _=toggle),
            status_cell_clickable(a_status, "/check_advancedhmc/$pn", detail_id),
            h.td(a_ess; class="u-pointer", _=toggle),
            h.td(a_time; class="u-pointer", _=toggle),
           ;
            id="row-$pn",
        ),
        h.tr(; id=detail_id, class="hidden")]
    end

    overview = h.div(
        h.h2("WarmupHMC PosteriorDB Dashboard ($(length(posterior_names)) posteriors)"),
        h.p(
            hx_link("/examples")("Examples (progress demo)"),
            " | ",
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
            h.tbody(reduce(vcat, [overview_row[pn] for pn in sort(posterior_names; by=pn -> !(@is_cached(compile_result[pn]) || @is_cached(sample_result[pn]) || @is_cached(dynamichmc_result[pn]) || @is_cached(advancedhmc_result[pn])))]; init=[])...; id="posterior-tbody")
        ),
        sortable_table_js(),
        h.style(".hidden { display: none; } tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    result_section(label, status, result) = begin
        status == :unstarted && return ""
        if status == :started
            return h.div(; class="u-mb-2")(
                h.p(h.strong(label, ": "), status_badge(:failed; label="FAIL")),
                h.p(h.em("Click the row's FAIL cell or re-run via the check route to view the error.")),
            )
        end
        h.div(; class="u-mb-2")(
            h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
            hasproperty(result, :dimension) && !isnothing(result.dimension) ? h.p(
                h.strong("Dimension: "), string(result.dimension)
            ) : "",
            hasproperty(result, :n_draws) && !isnothing(result.n_draws) ? h.p(
                h.strong("Draws: "), string(result.n_draws),
                " | ", h.strong("Min ESS: "), string(round(result.min_ess; digits=1)),
                " | ", h.strong("Median ESS: "), string(round(result.median_ess; digits=1)),
                " | ", h.strong("Time: "), string(round(result.time; digits=2)), "s",
                " | ", h.strong("Divergent: "), string(result.n_divergent)
            ) : "",
        )
    end

    reparam_section(pn) = begin
        has_reparam = !isnothing(posterior_reparametrization(pn))
        has_reparam || return ""
        r_status = reparam_status[pn]
        if r_status == :unstarted
            return h.div(; class="u-mb-2")(
                h.p(h.strong("Reparam: "),
                    h.a("Run"; hx_get=__self__/"check_reparam/$pn", hx_target="closest div", hx_swap="outerHTML",
                        class="u-pointer"))
            )
        end
        r_result = r_status == :ready ? reparam_result[pn] : nothing
        centering_info = if !isnothing(r_result) && !isnothing(r_result.centering)
            h.p(h.strong("Final centering: "),
                join(["[$idx] = $(round(c; digits=3))" for (idx, c) in r_result.centering[1:min(8,end)]], ", "),
                length(r_result.centering) > 8 ? ", ..." : "")
        else
            ""
        end
        h.div(; class="u-mb-2")(
            result_section["Reparam", r_status, r_result],
            centering_info,
        )
    end

    model_detail_content(pn) = begin
        c_status = compile_status[pn]
        s_status = sample_status[pn]
        d_status = dynamichmc_status[pn]
        a_status = advancedhmc_status[pn]
        c_result = c_status == :ready ? compile_result[pn] : nothing
        s_result = s_status == :ready ? sample_result[pn] : nothing
        d_result = d_status == :ready ? dynamichmc_result[pn] : nothing
        a_result = a_status == :ready ? advancedhmc_result[pn] : nothing
        statuses = (c_status, s_status, d_status, a_status)
        any_fail = any(==(:started), statuses)
        all_pass_or_unstarted = all(s -> s in (:ready, :unstarted), statuses)
        status_class = any_fail ? "u-status-callout u-status-error" : all_pass_or_unstarted ? "u-status-callout u-status-success" : "u-status-callout"
        h.td(; colspan="11", class="whmc-detail-cell")(
            h.div(; class=status_class)(
                h.h4(pn),
                result_section["Compiles", c_status, c_result],
                result_section["WarmupHMC", s_status, s_result],
                reparam_section[pn],
                result_section["DynamicHMC", d_status, d_result],
                result_section["AdvancedHMC", a_status, a_result],
            )
        )
    end

    updated_summary_row(pn) = begin
        c_status = compile_status[pn]
        s_status = sample_status[pn]
        d_status = dynamichmc_status[pn]
        a_status = advancedhmc_status[pn]
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"

        w_ess = s_status == :ready ? string(round(sample_result[pn].median_ess; digits=1)) : "-"
        w_time = s_status == :ready ? string(round(sample_result[pn].time; digits=2), "s") : "-"
        d_ess = d_status == :ready ? string(round(dynamichmc_result[pn].median_ess; digits=1)) : "-"
        d_time = d_status == :ready ? string(round(dynamichmc_result[pn].time; digits=2), "s") : "-"
        a_ess = a_status == :ready ? string(round(advancedhmc_result[pn].median_ess; digits=1)) : "-"
        a_time = a_status == :ready ? string(round(advancedhmc_result[pn].time; digits=2), "s") : "-"

        h.tr(
            h.td(pn; class="u-pointer", _=toggle),
            status_cell_clickable(c_status, "/check_compile/$pn", detail_id),
            status_cell_clickable(s_status, "/check_sample/$pn", detail_id),
            h.td(w_ess; class="u-pointer", _=toggle),
            h.td(w_time; class="u-pointer", _=toggle),
            status_cell_clickable(d_status, "/check_dynamichmc/$pn", detail_id),
            h.td(d_ess; class="u-pointer", _=toggle),
            h.td(d_time; class="u-pointer", _=toggle),
            status_cell_clickable(a_status, "/check_advancedhmc/$pn", detail_id),
            h.td(a_ess; class="u-pointer", _=toggle),
            h.td(a_time; class="u-pointer", _=toggle),
           ;
            id="row-$pn",
            hx_swap_oob="outerHTML:#row-$pn",
        )
    end

    sidebar_html = h.aside(; id="sidebar")(
        h.nav(
            h.div(h.strong("WarmupHMC"); class="sidebar-title"),
            h.ul(
                h.li(h.a("Table"; class="nav-item", data_nav="table",
                    hx_get=__self__/"fragment_table", hx_target="#content", hx_swap="innerHTML",
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
    force_check(check, pn) = if check == "compile"
        compile_status[pn] == :started && @clear_cache! compile_result[pn]
        compile_result[pn]
    elseif check == "sample"
        sample_status[pn] == :started && @clear_cache! sample_result[pn]
        sample_result[pn]
    elseif check == "dynamichmc"
        dynamichmc_status[pn] == :started && @clear_cache! dynamichmc_result[pn]
        dynamichmc_result[pn]
    elseif check == "advancedhmc"
        advancedhmc_status[pn] == :started && @clear_cache! advancedhmc_result[pn]
        advancedhmc_result[pn]
    elseif check == "reparam"
        reparam_status[pn] == :started && @clear_cache! reparam_result[pn]
        reparam_result[pn]
    end

    check_status(check, pn) = if check == "compile"
        compile_status[pn]
    elseif check == "sample"
        sample_status[pn]
    elseif check == "dynamichmc"
        dynamichmc_status[pn]
    elseif check == "advancedhmc"
        advancedhmc_status[pn]
    elseif check == "reparam"
        reparam_status[pn]
    else
        :unstarted
    end

    filtered_names(check, status_filter) = begin
        names = String[]
        for pn in posterior_names
            s = check_status[check, pn]
            show = if status_filter == "pass"; s == :ready
            elseif status_filter == "fail"; s == :started
            elseif status_filter == "unchecked"; s == :unstarted
            else; true
            end
            show && push!(names, pn)
        end
        names
    end

    @get serve_static(filename) = begin
        filepath = joinpath(static_dir(), filename)
        isfile(filepath) || return HTTP.Response(404, body="Not found")
        ext = lowercase(splitext(filename)[2])
        ct = ext == ".js" ? "application/javascript" : ext == ".css" ? "text/css" : "text/plain"
        HTTP.Response(200, ["Content-Type" => ct], body=read(filepath))
    end

    @get debug_reparam(pn) = begin
        reparam = posterior_reparametrization(pn)
        isnothing(reparam) && return "No reparametrization for $pn"
        problem = stan_problem(pn)
        dim = LogDensityProblems.dimension(problem)
        x = randn(Xoshiro(42), dim)
        lines = String[]
        push!(lines, "posterior: $pn (dim=$dim)")
        push!(lines, "x = $x")
        ljac, y = reparam(x)
        push!(lines, "ljac = $ljac")
        push!(lines, "y = $y")
        ld = LogDensityProblems.logdensity(problem, y)
        push!(lines, "logdensity(problem, y) = $ld")
        rp = ReparametrizedProblem(reparam, problem, AutoForwardDiff())
        ld_rp = LogDensityProblems.logdensity(rp, x)
        push!(lines, "logdensity(rp, x) = $ld_rp")
        push!(lines, "capabilities(rp) = $(LogDensityProblems.capabilities(typeof(rp)))")
        ld_rp2, g = LogDensityProblems.logdensity_and_gradient(rp, x)
        push!(lines, "logdensity_and_gradient(rp, x) = ($ld_rp2, $(g[1:min(3,end)])...)")
        join(lines, "\n")
    end

    @get pkg_resolve = begin
        Pkg.resolve()
        Pkg.instantiate()
        "Pkg.resolve() and Pkg.instantiate() completed"
    end

    @get index = overview

    @get model(pn) = h.div(; id="content")(
        breadcrumb([
            ("Table", "/fragment_table", "/"),
            (pn, nothing, nothing),
        ]),
        model_detail_content[pn],
    )

    @get check_compile(pn) = begin
        force_check["compile", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_sample(pn) = begin
        force_check["sample", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_dynamichmc(pn) = begin
        force_check["dynamichmc", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_advancedhmc(pn) = begin
        force_check["advancedhmc", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_reparam(pn) = begin
        reparam_result[pn]
        reparam_section[pn]
    end

    @get clear(check, pn) = begin
        if check == "compile"
            @clear_cache! compile_result[pn]
        elseif check == "sample"
            @clear_cache! sample_result[pn]
        elseif check == "dynamichmc"
            @clear_cache! dynamichmc_result[pn]
        elseif check == "advancedhmc"
            @clear_cache! advancedhmc_result[pn]
        elseif check == "reparam"
            @clear_cache! reparam_result[pn]
        end
        "Cleared $check cache for $pn"
    end

    @get filter(check, status="all") = join(filtered_names[check, status], "\n")

    @get recheck(check, status="fail") = begin
        targets = filtered_names[check, status]
        results = String[]
        for pn in targets
            force_check[check, pn]
            push!(results, "PASS $pn")
        end
        join(results, "\n")
    end

    # --- Examples with progress ---
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

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
end

function __init__()
    route!(AppContext())
end

end # module WarmupHMCWeb
