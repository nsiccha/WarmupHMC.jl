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
using ReactiveObjects, ReactiveHMC, ElasticArrays
import KernelDensity
using TestModules

include("test/runtests.jl")

pdb = PosteriorDB.database()

static_dir() = joinpath(dirname(@__DIR__), "static")

viz_css() = """
.mcmc-grid {
  display: grid; gap: 6px;
  grid-template-columns: 2fr 1fr;
  grid-template-rows: repeat(4, 110px);
  grid-template-areas: "viz hist-y" "viz trace-y" "viz hist-x" "viz trace-x";
}
.panel { background: #fff; border: 1px solid #ddd; border-radius: 2px; overflow: hidden; }
.viz { grid-area: viz; } .hist-y { grid-area: hist-y; } .hist-x { grid-area: hist-x; }
.trace-x { grid-area: trace-x; } .trace-y { grid-area: trace-y; }
.panel-label { font-size: 0.7rem; color: #888; padding: 4px 0; }
.shared-trace { flex: 1; height: 80px; background: #fff; border: 1px solid #ddd; border-radius: 2px; overflow: hidden; }
.speed-display { font-size: 0.6rem; color: #888; font-family: monospace; min-width: 52px; text-align: center; }
#progress-bar { cursor: pointer; display: block; }
.sampler-btn { cursor: pointer; margin-right: 8px; }
.sampler-btn.active { font-weight: bold; opacity: 1 !important; }
"""

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

function try_compile(posterior_name)
    try
        problem = stan_problem(posterior_name)
        dim = LogDensityProblems.dimension(problem)
        (ok=true, error=nothing, stacktrace=nothing, dimension=dim)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace, dimension=nothing)
    end
end

function try_sample(posterior_name; n_draws=100, seed=42)
    try
        problem = stan_problem(posterior_name)
        dim = LogDensityProblems.dimension(problem)
        rng = Xoshiro(seed)
        t0 = time()
        result = adaptive_warmup_mcmc(rng, problem; n_draws)
        elapsed = time() - t0
        # result returns posterior_position (dim × n_draws)
        draws = result.posterior_position
        n = size(draws, 2)
        # Compute ESS ourselves since monitor_ess requires progress
        dim = size(draws, 1)
        ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
        min_ess = minimum(ess_vals)
        median_ess = median(ess_vals)
        draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
        (ok=true, error=nothing, stacktrace=nothing,
         n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
         n_divergent=result.n_divergent_samples, draws_2d=draws_2d)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace,
         n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
         n_divergent=nothing, draws_2d=nothing)
    end
end

function try_sample_reparam(posterior_name; n_draws=100, seed=42)
    reparam = posterior_reparametrization(posterior_name)
    isnothing(reparam) && return (ok=false, error="No reparametrization defined for $posterior_name", stacktrace=nothing,
        n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
        n_divergent=nothing, draws_2d=nothing, centering=nothing)
    try
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
        # Extract final centering values
        centering = [(idx, v.source.c) for (idx, v) in reparam.pairs]
        (ok=true, error=nothing, stacktrace=nothing,
         n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
         n_divergent=result.n_divergent_samples, draws_2d=draws_2d, centering=centering)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace,
         n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
         n_divergent=nothing, draws_2d=nothing, centering=nothing)
    end
end

function try_sample_dynamichmc(posterior_name; n_draws=100, seed=42)
    try
        problem = stan_problem(posterior_name)
        dim = LogDensityProblems.dimension(problem)
        rng = Xoshiro(seed)
        t0 = time()
        result = WarmupHMC.DynamicHMC.mcmc_with_warmup(rng, problem, n_draws; reporter=WarmupHMC.DynamicHMC.NoProgressReport())
        elapsed = time() - t0
        draws = result.posterior_matrix  # dim × n_draws
        n = size(draws, 2)
        ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
        min_ess = minimum(ess_vals)
        median_ess = median(ess_vals)
        n_divergent = count(s -> WarmupHMC.DynamicHMC.is_divergent(s.termination), result.tree_statistics)
        draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
        (ok=true, error=nothing, stacktrace=nothing,
         n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
         n_divergent=n_divergent, draws_2d=draws_2d)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace,
         n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
         n_divergent=nothing, draws_2d=nothing)
    end
end

function try_sample_advancedhmc(posterior_name; n_draws=100, n_adapts=100, seed=42)
    try
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
        draws = reduce(hcat, θs)  # dim × n_draws
        n = size(draws, 2)
        ess_vals = MCMCDiagnosticTools.ess(reshape(draws', (:, 1, dim)))
        min_ess = minimum(ess_vals)
        median_ess = median(ess_vals)
        n_divergent = sum(s.numerical_error for s in stats)
        draws_2d = dim >= 2 ? [[draws[1,i], draws[2,i]] for i in 1:n] : [[draws[1,i], 0.0] for i in 1:n]
        (ok=true, error=nothing, stacktrace=nothing,
         n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
         n_divergent=n_divergent, draws_2d=draws_2d)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace,
         n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
         n_divergent=nothing, draws_2d=nothing)
    end
end

# --- ReactiveHMC trajectory recording ---

# Potential energy functions (negated log density) for ReactiveHMC
pot(problem, x) = -LogDensityProblems.logdensity(problem, x)
pot_and_grad(problem, x) = .-LogDensityProblems.logdensity_and_gradient(problem, x)

# Partial function application (matches LocalScalesHMC's partial)
struct StepFn{F} <: Function
    f::F
    stepsize::Float64
end
(s::StepFn)(args...) = s.f(args...; stepsize=s.stepsize)

# trajectory_stats and sampling_stats are now provided by ReactiveHMC

invperm0(x) = invperm(x .+ 1) .- 1

warmup_strategies() = ["none", "stan", "stan_win", "nutpie", "nutpie_win"]
warmup_label(w) = Dict(
    "none" => "DA only",
    "stan" => "DA + position metric",
    "stan_win" => "stan windowed",
    "nutpie" => "DA + pos/grad metric",
    "nutpie_win" => "nutpie windowed",
)[w]

# Compute metric diagonal from welford accumulators
function metric_diag_stan(wvp)
    max.(1e-6, wvp.var)
end
function metric_diag_nutpie(wvp, wvg)
    # sqrt(var_pos / var_grad) — equivalent to (var_pos/var_grad)^0.5
    max.(1e-6, sqrt.(max.(1e-12, wvp.var) ./ max.(1e-12, wvg.var)))
end

function try_sample_reactive(posterior_name; warmup="none", n_draws=100, n_adapts=50, seed=42, progress=nothing)
    try
        update_progress!(progress, "Compiling Stan model...")
        problem = stan_problem(posterior_name)
        dim = LogDensityProblems.dimension(problem)
        update_progress!(progress, "Compiled (dim=$dim). Starting sampler...")
        rng = Xoshiro(seed)

        pot_f = Base.Fix1(pot, problem)
        grad_f = Base.Fix1(pot_and_grad, problem)

        init_pos = randn(rng, dim)
        init_mom = zeros(dim)
        metric = Diagonal(ones(dim))
        phasepoint = euclidean_phasepoint(pot_f, grad_f, metric, init_pos, init_mom)
        tstats = trajectory_stats(dim)

        da_state = dual_averaging_state(1.0; target=0.8)
        state = nuts_state(phasepoint; rng, step_f=StepFn(leapfrog!, 1.0), stats_f=tstats)

        # --- Metric adaptation setup ---
        use_metric = warmup in ("stan", "stan_win", "nutpie", "nutpie_win")
        use_grads = warmup in ("nutpie", "nutpie_win")
        use_windowed = warmup in ("stan_win", "nutpie_win")

        # Foreground estimators
        wvp_fg = use_metric ? welford_var(dim) : nothing
        wvg_fg = use_grads ? welford_var(dim) : nothing
        # Background estimators (nutpie windowed only)
        wvp_bg = use_windowed ? welford_var(dim) : nothing
        wvg_bg = use_windowed ? welford_var(dim) : nothing
        bg_count = 0

        # nutpie windowed phase boundaries
        early_end = floor(Int, 0.3 * n_adapts)
        final_ss_start = n_adapts - floor(Int, 0.15 * n_adapts)
        early_switch_freq = 10
        mid_switch_freq = 80

        # Accumulator (from ReactiveHMC)
        dstats = sampling_stats(tstats)

        t0 = time()
        total_draws = n_draws + n_adapts
        pnode = initialize_progress!(progress, total_draws; description="MCMC ($warmup)")
        for i in 1:total_draws
            reset!(tstats, state.init)
            @invalidatedependants! state.init.mom = sqrt(state.init.metric) * randn!(rng, state.init.mom)
            step!(state)

            # Collect stats
            dstats(state, da_state)
            update_progress!(pnode, i;
                phase=i < n_adapts ? "warmup" : "sampling",
                stepsize=short_string(state.step_f.stepsize),
                acc_rate=short_string(Fraction(dstats.acc_rate[end])),
            )

            if i < n_adapts
                # Step size adaptation (always)
                fit!(da_state, dstats.acc_rate[end])
                state.step_f = StepFn(leapfrog!, da_state.current)

                if use_metric && !use_windowed
                    # Simple online metric: stan or nutpie (no windowing)
                    step!(wvp_fg, state.init.pos)
                    use_grads && step!(wvg_fg, state.init.dpot_dpos)
                    if wvp_fg.n > 2
                        new_diag = use_grads && wvg_fg.n > 2 ?
                            metric_diag_nutpie(wvp_fg, wvg_fg) :
                            metric_diag_stan(wvp_fg)
                        @invalidatedependants! state.init.metric = Diagonal(new_diag)
                        da_state = dual_averaging_state(state.step_f.stepsize; target=0.8)
                    end

                elseif use_windowed && i <= final_ss_start
                    # Windowed adaptation: feed foreground + background estimators
                    pos = copy(state.init.pos)
                    step!(wvp_fg, pos)
                    step!(wvp_bg, pos)
                    if use_grads
                        grad = copy(state.init.dpot_dpos)
                        step!(wvg_fg, grad)
                        step!(wvg_bg, grad)
                    end
                    bg_count += 1

                    # Switch foreground <- background at the right frequency
                    switch_freq = i <= early_end ? early_switch_freq : mid_switch_freq
                    if bg_count >= switch_freq
                        wvp_fg = wvp_bg
                        wvp_bg = welford_var(dim)
                        if use_grads
                            wvg_fg = wvg_bg
                            wvg_bg = welford_var(dim)
                        end
                        bg_count = 0
                    end

                    # Update metric every draw
                    if wvp_fg.n > 2
                        new_diag = use_grads && wvg_fg.n > 2 ?
                            metric_diag_nutpie(wvp_fg, wvg_fg) :
                            metric_diag_stan(wvp_fg)
                        @invalidatedependants! state.init.metric = Diagonal(new_diag)
                        da_state = dual_averaging_state(state.step_f.stepsize; target=0.8)
                    end
                end
                # nutpie_win late phase (i > final_ss_start): metric frozen, only DA continues

            elseif i == n_adapts
                state.step_f = StepFn(leapfrog!, da_state.final)
            end
        end
        finalize_progress!(pnode)
        elapsed = time() - t0

        post_draws = dstats.draws[:, (n_adapts+1):end]
        n = size(post_draws, 2)
        ess_vals = MCMCDiagnosticTools.ess(reshape(post_draws', (:, 1, dim)))
        min_ess = minimum(ess_vals)
        median_ess = median(ess_vals)
        n_divergent = sum(dstats.diverged[(n_adapts+1):end])

        (ok=true, error=nothing, stacktrace=nothing,
         n_draws=n, dimension=dim, min_ess=min_ess, median_ess=median_ess, time=elapsed,
         n_divergent=n_divergent,
         draws_2d=nothing,
         full_history=dstats.full_history,
         full_idxs=dstats.full_idxs,
         all_draws=Matrix(dstats.draws),
         n_adapts=n_adapts,
         ess_vals=vec(ess_vals),
         stepsizes=dstats.stepsizes,
         acc_rate=dstats.acc_rate,
         all_n_steps=dstats.n_steps,
         all_diverged=dstats.diverged)
    catch e
        bt = catch_backtrace()
        error_msg = first(split(sprint(showerror, e), "\n"))
        full_trace = sprint(showerror, e, bt)
        (ok=false, error=error_msg, stacktrace=full_trace,
         n_draws=nothing, dimension=nothing, min_ess=nothing, median_ess=nothing, time=nothing,
         n_divergent=nothing, draws_2d=nothing,
         full_history=nothing, full_idxs=nothing, all_draws=nothing, n_adapts=nothing,
         ess_vals=nothing, stepsizes=nothing, acc_rate=nothing, all_n_steps=nothing,
         all_diverged=nothing)
    end
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


function status_str(cached, ok)
    !cached ? "-" : ok ? "PASS" : "FAIL"
end

function status_cell(cached, ok)
    !cached && return h.td("-"; style="color:gray")
    ok ? h.td("PASS"; style="color:green;font-weight:bold") :
         h.td("FAIL"; style="color:red;font-weight:bold")
end

function status_cell_clickable(cached, ok, check_url, detail_id)
    base_style = "cursor:pointer;"
    if !cached
        return h.td("-"; class="check-cell", style=base_style * "color:gray",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    elseif !ok
        return h.td("FAIL"; class="check-cell", style=base_style * "color:red;font-weight:bold",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    else
        return h.td("PASS"; class="check-cell", style=base_style * "color:green;font-weight:bold",
            _="on click toggle .hidden on #$detail_id")
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

    pathfinder[pn] = run_parallel_pathfinder(pn; progress=__status__)
    sampling[pn] = run_parallel_sampling(pn; progress=__status__)
    cmdstan[pn] = run_cmdstan_sampling(pn; progress=__status__)
end
_examples = ExampleComputations(; cache_type=:parallel)


function _posterior_select(names, id)
    h.select(; id, name="pn")(
        [h.option(pn; value=pn) for pn in names]...
    )
end


# --- Async reactive sampling ---
@dynamicstruct struct AsyncReactiveComputations
    __status__ = initialize_progress!(:state; description="Reactive")

    results(pn, w) = try_sample_reactive(pn; warmup=w, progress=__status__)
end
_async_reactive = AsyncReactiveComputations(; cache_type=:parallel)

@htmx struct AppContext
    
    cache_path = joinpath(dirname(dirname(@__DIR__)), "web", "cache")

    @cached posterior_names = sort([
        pn for pn in PosteriorDB.posterior_names(pdb)
        if !isnothing(
            try
                PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, pn)), "stan")
            catch
                nothing
            end
        )
    ])

    @cached compile_result(pn) = try_compile(pn)

    @cached sample_result(pn) = try_sample(pn)

    @cached dynamichmc_result(pn) = try_sample_dynamichmc(pn)

    @cached advancedhmc_result(pn) = try_sample_advancedhmc(pn)

    @cached reparam_result(pn) = try_sample_reparam(pn)

    @cached reactive_result(pn, warmup) = try_sample_reactive(pn; warmup)

    overview_row(pn) = begin
        c_cached = @is_cached compile_result[pn]
        c_ok = c_cached ? compile_result[pn].ok : false
        s_cached = @is_cached sample_result[pn]
        s_ok = s_cached ? sample_result[pn].ok : false
        d_cached = @is_cached dynamichmc_result[pn]
        d_ok = d_cached ? dynamichmc_result[pn].ok : false
        a_cached = @is_cached advancedhmc_result[pn]
        a_ok = a_cached ? advancedhmc_result[pn].ok : false
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"

        w_ess = s_cached && s_ok ? string(round(sample_result[pn].median_ess; digits=1)) : "-"
        w_time = s_cached && s_ok ? string(round(sample_result[pn].time; digits=2), "s") : "-"
        d_ess = d_cached && d_ok ? string(round(dynamichmc_result[pn].median_ess; digits=1)) : "-"
        d_time = d_cached && d_ok ? string(round(dynamichmc_result[pn].time; digits=2), "s") : "-"
        a_ess = a_cached && a_ok ? string(round(advancedhmc_result[pn].median_ess; digits=1)) : "-"
        a_time = a_cached && a_ok ? string(round(advancedhmc_result[pn].time; digits=2), "s") : "-"

        [h.tr(
            h.td(pn; style="cursor:pointer", _=toggle),
            status_cell_clickable(c_cached, c_ok, "/check_compile/$pn", detail_id),
            status_cell_clickable(s_cached, s_ok, "/check_sample/$pn", detail_id),
            h.td(w_ess; style="cursor:pointer", _=toggle),
            h.td(w_time; style="cursor:pointer", _=toggle),
            status_cell_clickable(d_cached, d_ok, "/check_dynamichmc/$pn", detail_id),
            h.td(d_ess; style="cursor:pointer", _=toggle),
            h.td(d_time; style="cursor:pointer", _=toggle),
            status_cell_clickable(a_cached, a_ok, "/check_advancedhmc/$pn", detail_id),
            h.td(a_ess; style="cursor:pointer", _=toggle),
            h.td(a_time; style="cursor:pointer", _=toggle),
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
            h.a(href="/tests")("Tests"),
            " | ",
            hx_link("/viz_picker")("Viz"),
        ),
        h.input(;
            type="search",
            id="search",
            placeholder="Filter posteriors...",
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove .hidden from row else add .hidden to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not(.hidden)' set target to null for cell in <td.check-cell/> in row if target is null and cell.textContent.trim() is not 'PASS' set target to cell end end if target is not null add .batch to target send click to target end end end",
            style="margin-bottom:1rem;width:100%;",
        ),
        h.table(class="striped"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)", style="cursor:pointer", rowspan="2"),
                    h.th("Compiles"; _="on click call sortTable(1, me)", style="cursor:pointer", rowspan="2"),
                    h.th("WarmupHMC"; colspan="3", style="text-align:center;border-bottom:none"),
                    h.th("DynamicHMC"; colspan="3", style="text-align:center;border-bottom:none"),
                    h.th("AdvancedHMC"; colspan="3", style="text-align:center;border-bottom:none"),
                ),
                h.tr(
                    h.th("Status"; _="on click call sortTable(2, me)", style="cursor:pointer"),
                    h.th("ESS"; _="on click call sortTable(3, me)", style="cursor:pointer"),
                    h.th("Time"; _="on click call sortTable(4, me)", style="cursor:pointer"),
                    h.th("Status"; _="on click call sortTable(5, me)", style="cursor:pointer"),
                    h.th("ESS"; _="on click call sortTable(6, me)", style="cursor:pointer"),
                    h.th("Time"; _="on click call sortTable(7, me)", style="cursor:pointer"),
                    h.th("Status"; _="on click call sortTable(8, me)", style="cursor:pointer"),
                    h.th("ESS"; _="on click call sortTable(9, me)", style="cursor:pointer"),
                    h.th("Time"; _="on click call sortTable(10, me)", style="cursor:pointer"),
                ),
            ),
            h.tbody(reduce(vcat, [overview_row[pn] for pn in sort(posterior_names; by=pn -> !(@is_cached(compile_result[pn]) || @is_cached(sample_result[pn]) || @is_cached(dynamichmc_result[pn]) || @is_cached(advancedhmc_result[pn])))]; init=[])...; id="posterior-tbody")
        ),
        sortable_table_js(),
        h.style(".hidden { display: none; } tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    result_section(label, result) = begin
        isnothing(result) && return ""
        h.div(; style="margin-bottom:0.5rem")(
            h.p(h.strong(label, ": "), status_badge(result.ok ? :done : :failed; label=result.ok ? "PASS" : "FAIL")),
            isnothing(result.error) ? "" : h.p(h.strong("Error: "), h.code(result.error)),
            isnothing(result.stacktrace) ? "" : h.details(
                h.summary("Full stacktrace"),
                h.pre(result.stacktrace; style="white-space:pre-wrap;max-height:300px;overflow:auto;font-size:0.8em")
            ),
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
        r_cached = @is_cached reparam_result[pn]
        r_result = r_cached ? reparam_result[pn] : nothing
        has_reparam = !isnothing(posterior_reparametrization(pn))
        if !has_reparam
            ""
        elseif isnothing(r_result)
            h.div(; style="margin-bottom:0.5rem")(
                h.p(h.strong("Reparam: "),
                    h.a("Run"; hx_get="/check_reparam/$pn", hx_target="closest div", hx_swap="outerHTML",
                        style="cursor:pointer"))
            )
        else
            centering_info = if r_result.ok && !isnothing(r_result.centering)
                h.p(h.strong("Final centering: "),
                    join(["[$idx] = $(round(c; digits=3))" for (idx, c) in r_result.centering[1:min(8,end)]], ", "),
                    length(r_result.centering) > 8 ? ", ..." : "")
            else
                ""
            end
            h.div(; style="margin-bottom:0.5rem")(
                result_section["Reparam", r_result],
                centering_info,
            )
        end
    end

    model_detail_content(pn) = begin
        c_cached = @is_cached compile_result[pn]
        s_cached = @is_cached sample_result[pn]
        d_cached = @is_cached dynamichmc_result[pn]
        a_cached = @is_cached advancedhmc_result[pn]
        c_result = c_cached ? compile_result[pn] : nothing
        s_result = s_cached ? sample_result[pn] : nothing
        d_result = d_cached ? dynamichmc_result[pn] : nothing
        a_result = a_cached ? advancedhmc_result[pn] : nothing
        all_results = [c_result, s_result, d_result, a_result]
        all_ok = all(r -> isnothing(r) || r.ok, all_results)
        any_fail = any(r -> !isnothing(r) && !r.ok, all_results)
        border_color = any_fail ? "var(--pico-del-color)" : all_ok ? "var(--pico-ins-color)" : "var(--pico-muted-border-color)"
        h.td(; colspan="11", style="padding:0;border:none")(
            h.div(; style="padding:1rem 1.5rem;background:var(--pico-card-background-color);border-left:4px solid $border_color;margin:0.25rem 0")(
                h.h4(pn, " ", h.a("▶ Viz";
                    hx_get="/fragment_viz/$pn", hx_target="#content", hx_swap="innerHTML",
                    hx_push_url="/viz/$pn",
                    style="font-size:0.8em;font-weight:normal;cursor:pointer")),
                result_section["Compiles", c_result],
                result_section["WarmupHMC", s_result],
                reparam_section[pn],
                result_section["DynamicHMC", d_result],
                result_section["AdvancedHMC", a_result],
            )
        )
    end

    updated_summary_row(pn) = begin
        c_cached = @is_cached compile_result[pn]
        c_ok = c_cached ? compile_result[pn].ok : false
        s_cached = @is_cached sample_result[pn]
        s_ok = s_cached ? sample_result[pn].ok : false
        d_cached = @is_cached dynamichmc_result[pn]
        d_ok = d_cached ? dynamichmc_result[pn].ok : false
        a_cached = @is_cached advancedhmc_result[pn]
        a_ok = a_cached ? advancedhmc_result[pn].ok : false
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"

        w_ess = s_cached && s_ok ? string(round(sample_result[pn].median_ess; digits=1)) : "-"
        w_time = s_cached && s_ok ? string(round(sample_result[pn].time; digits=2), "s") : "-"
        d_ess = d_cached && d_ok ? string(round(dynamichmc_result[pn].median_ess; digits=1)) : "-"
        d_time = d_cached && d_ok ? string(round(dynamichmc_result[pn].time; digits=2), "s") : "-"
        a_ess = a_cached && a_ok ? string(round(advancedhmc_result[pn].median_ess; digits=1)) : "-"
        a_time = a_cached && a_ok ? string(round(advancedhmc_result[pn].time; digits=2), "s") : "-"

        h.tr(
            h.td(pn; style="cursor:pointer", _=toggle),
            status_cell_clickable(c_cached, c_ok, "/check_compile/$pn", detail_id),
            status_cell_clickable(s_cached, s_ok, "/check_sample/$pn", detail_id),
            h.td(w_ess; style="cursor:pointer", _=toggle),
            h.td(w_time; style="cursor:pointer", _=toggle),
            status_cell_clickable(d_cached, d_ok, "/check_dynamichmc/$pn", detail_id),
            h.td(d_ess; style="cursor:pointer", _=toggle),
            h.td(d_time; style="cursor:pointer", _=toggle),
            status_cell_clickable(a_cached, a_ok, "/check_advancedhmc/$pn", detail_id),
            h.td(a_ess; style="cursor:pointer", _=toggle),
            h.td(a_time; style="cursor:pointer", _=toggle),
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
                    hx_get="/fragment_table", hx_target="#content", hx_swap="innerHTML",
                    hx_push_url="/",
                    _="on click remove .nav-active from .nav-item then add .nav-active to me")),
                h.li(h.a("Viz"; class="nav-item", data_nav="viz",
                    hx_get="/fragment_viz_picker", hx_target="#content", hx_swap="innerHTML",
                    hx_push_url="false",
                    _="on click remove .nav-active from .nav-item then add .nav-active to me")),
            ),
        ),
    )

    page(content) = htmx(
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
        """),
        h.script(raw"""
            function updateNav() {
                var path = window.location.pathname;
                document.querySelectorAll('.nav-item').forEach(function(el) { el.classList.remove('nav-active'); });
                if (path.startsWith('/viz') || path.startsWith('/check_reactive')) {
                    var el = document.querySelector('[data-nav="viz"]');
                    if (el) el.classList.add('nav-active');
                } else {
                    var el = document.querySelector('[data-nav="table"]');
                    if (el) el.classList.add('nav-active');
                }
            }
            document.addEventListener('DOMContentLoaded', updateNav);
            document.body.addEventListener('htmx:pushedIntoHistory', updateNav);
        """);
        pico_version="2",
    )

    # Force (re)compute a single check, clearing failed cache first
    force_check(check, pn) = if check == "compile"
        @is_cached(compile_result[pn]) && !compile_result[pn].ok && @clear_cache! compile_result[pn]
        compile_result[pn]
    elseif check == "sample"
        @is_cached(sample_result[pn]) && !sample_result[pn].ok && @clear_cache! sample_result[pn]
        sample_result[pn]
    elseif check == "dynamichmc"
        @is_cached(dynamichmc_result[pn]) && !dynamichmc_result[pn].ok && @clear_cache! dynamichmc_result[pn]
        dynamichmc_result[pn]
    elseif check == "advancedhmc"
        @is_cached(advancedhmc_result[pn]) && !advancedhmc_result[pn].ok && @clear_cache! advancedhmc_result[pn]
        advancedhmc_result[pn]
    elseif check == "reparam"
        @is_cached(reparam_result[pn]) && !reparam_result[pn].ok && @clear_cache! reparam_result[pn]
        reparam_result[pn]
    end

    check_status(check, pn) = if check == "compile"
        c = @is_cached compile_result[pn]
        (c, c ? compile_result[pn].ok : false)
    elseif check == "sample"
        c = @is_cached sample_result[pn]
        (c, c ? sample_result[pn].ok : false)
    elseif check == "dynamichmc"
        c = @is_cached dynamichmc_result[pn]
        (c, c ? dynamichmc_result[pn].ok : false)
    elseif check == "advancedhmc"
        c = @is_cached advancedhmc_result[pn]
        (c, c ? advancedhmc_result[pn].ok : false)
    elseif check == "reparam"
        c = @is_cached reparam_result[pn]
        (c, c ? reparam_result[pn].ok : false)
    else
        (false, false)
    end

    filtered_names(check, status) = begin
        names = String[]
        for pn in posterior_names
            cached, ok = check_status[check, pn]
            show = if status == "pass"; cached && ok
            elseif status == "fail"; cached && !ok
            elseif status == "unchecked"; !cached
            else; true
            end
            show && push!(names, pn)
        end
        names
    end

    plain_overview = begin
        header = rpad("Posterior", 45) * rpad("Compile", 9) * "| " * rpad("WarmupHMC", 10) * rpad("ESS", 10) * rpad("Time", 10) * "| " * rpad("DynHMC", 10) * rpad("ESS", 10) * rpad("Time", 10) * "| " * rpad("AdvHMC", 10) * rpad("ESS", 10) * "Time"
        lines = [header, "-"^length(header)]
        for pn in sort(posterior_names; by=pn -> !(@is_cached(compile_result[pn]) || @is_cached(sample_result[pn]) || @is_cached(dynamichmc_result[pn]) || @is_cached(advancedhmc_result[pn])))
            c_cached = @is_cached compile_result[pn]
            c_ok = c_cached ? compile_result[pn].ok : false
            s_cached = @is_cached sample_result[pn]
            s_ok = s_cached ? sample_result[pn].ok : false
            d_cached = @is_cached dynamichmc_result[pn]
            d_ok = d_cached ? dynamichmc_result[pn].ok : false
            a_cached = @is_cached advancedhmc_result[pn]
            a_ok = a_cached ? advancedhmc_result[pn].ok : false
            w_ess = s_cached && s_ok ? string(round(sample_result[pn].median_ess; digits=1)) : "-"
            w_time = s_cached && s_ok ? string(round(sample_result[pn].time; digits=2)) : "-"
            d_ess = d_cached && d_ok ? string(round(dynamichmc_result[pn].median_ess; digits=1)) : "-"
            d_time = d_cached && d_ok ? string(round(dynamichmc_result[pn].time; digits=2)) : "-"
            a_ess = a_cached && a_ok ? string(round(advancedhmc_result[pn].median_ess; digits=1)) : "-"
            a_time = a_cached && a_ok ? string(round(advancedhmc_result[pn].time; digits=2)) : "-"
            push!(lines, rpad(pn, 45) * rpad(status_str(c_cached, c_ok), 9) * "| " * rpad(status_str(s_cached, s_ok), 10) * rpad(w_ess, 10) * rpad(w_time, 10) * "| " * rpad(status_str(d_cached, d_ok), 10) * rpad(d_ess, 10) * rpad(d_time, 10) * "| " * rpad(status_str(a_cached, a_ok), 10) * rpad(a_ess, 10) * a_time)
        end
        join(lines, "\n")
    end

    plain_result_section(label, result) = begin
        isnothing(result) && return "$label: -"
        parts = ["$label: $(result.ok ? "PASS" : "FAIL")"]
        isnothing(result.error) || push!(parts, "  Error: $(result.error)")
        hasproperty(result, :stacktrace) && !isnothing(result.stacktrace) && push!(parts, "", "  --- Stacktrace ---", result.stacktrace)
        hasproperty(result, :dimension) && !isnothing(result.dimension) && push!(parts, "  Dimension: $(result.dimension)")
        hasproperty(result, :n_draws) && !isnothing(result.n_draws) && push!(parts, "  Draws: $(result.n_draws), Min ESS: $(result.min_ess), Median ESS: $(result.median_ess), Time: $(result.time)s, Divergent: $(result.n_divergent)")
        join(parts, "\n")
    end

    plain_model(pn) = begin
        c_cached = @is_cached compile_result[pn]
        s_cached = @is_cached sample_result[pn]
        d_cached = @is_cached dynamichmc_result[pn]
        a_cached = @is_cached advancedhmc_result[pn]
        c_result = c_cached ? compile_result[pn] : nothing
        s_result = s_cached ? sample_result[pn] : nothing
        d_result = d_cached ? dynamichmc_result[pn] : nothing
        a_result = a_cached ? advancedhmc_result[pn] : nothing
        parts = ["# $pn", "",
            plain_result_section["Compiles", c_result], "",
            plain_result_section["WarmupHMC", s_result], "",
            plain_result_section["DynamicHMC", d_result], "",
            plain_result_section["AdvancedHMC", a_result],
        ]
        join(parts, "\n")
    end

    @get serve_static(filename) = begin
        filepath = joinpath(static_dir(), filename)
        isfile(filepath) || return HTTP.Response(404, body="Not found")
        ext = lowercase(splitext(filename)[2])
        ct = ext == ".js" ? "application/javascript" : ext == ".css" ? "text/css" : "text/plain"
        HTTP.Response(200, ["Content-Type" => ct], body=read(filepath))
    end

    viz_dataset(r, dims) = begin
        i, j = dims
        draws = r.all_draws
        full_history = r.full_history
        full_idxs = r.full_idxs
        n_adapts = r.n_adapts

        trajectories = [collect.(eachcol(t[[i,j], :])) for t in full_history]
        post_draws = draws[:, (n_adapts+1):end]
        kde_draws = post_draws[[i,j], :]

        xGrid = collect(range(extrema(kde_draws[1, :])..., 200))
        yGrid = collect(range(extrema(kde_draws[2, :])..., 200))
        kde2d = KernelDensity.kde(kde_draws')
        pdfs = max.(0, KernelDensity.pdf(KernelDensity.InterpKDE(kde2d), xGrid, yGrid))
        xPdf = collect(KernelDensity.pdf(KernelDensity.InterpKDE(KernelDensity.kde(kde_draws[1, :])), xGrid))
        yPdf = collect(KernelDensity.pdf(KernelDensity.InterpKDE(KernelDensity.kde(kde_draws[2, :])), yGrid))

        Dict(
            "xGrid" => xGrid, "yGrid" => yGrid,
            "pdfs" => collect.(eachrow(pdfs)),
            "trajectories" => trajectories,
            "xPdf" => xPdf, "yPdf" => yPdf,
        ), invperm0.(full_idxs)
    end

    viz_init_script(pn, w) = begin
        r = @is_cached(reactive_result[pn, w]) ? reactive_result[pn, w] : nothing
        (isnothing(r) || !r.ok) && return ""

        dim = r.dimension
        ess = r.ess_vals
        j, i = length(ess) >= 2 ? sortperm(ess) : (1, min(2, length(ess)))
        if i == j; i = min(2, dim); end

        data, order = viz_dataset[r, (i, j)]

        shared_specs = [
            Dict("label" => "step size", "logScale" => true, "traces" => [
                Dict("values" => r.stepsizes, "label" => "current"),
            ]),
            Dict("label" => "acceptance rate", "traces" => [
                Dict("values" => r.acc_rate, "label" => "current"),
                Dict("values" => cumsum(r.acc_rate) ./ eachindex(r.acc_rate), "label" => "cumulative"),
            ]),
            Dict("label" => "cumulative steps", "traces" => [
                Dict("values" => cumsum(r.all_n_steps), "label" => "total"),
            ]),
        ]

        options = Dict(
            "trajectories_order" => order,
            "zoomToFit" => [true],
        )

        data_json = JSON.json([data]; allownan=true)
        traces_json = JSON.json(shared_specs; allownan=true)
        options_json = JSON.json(options)

        """<script>setup_viz($data_json, $traces_json, $options_json).play()</script>"""
    end

    # --- Async reactive sampling with progress polling (fetchindex pattern) ---

    @get async_reactive(pn, w; force::Bool=false) = fetchindex(_async_reactive.results, pn, w; force) do rv, status
        if rv isa Task && istaskfailed(rv)
            h.article(h.header("Failed"), h.pre(sprint(showerror, rv.result)))
        elseif rv isa Task
            h.div(; hx_get=query_url("/async_reactive/$pn/$w"), hx_trigger="every 200ms", hx_swap="outerHTML")(
                h.article(
                    h.header("Sampling $pn ($(warmup_label(w)))..."),
                    htmx_render_children(status),
                )
            )
        else
            result_section[warmup_label(w), rv]
        end
    end

    @get viz(pn) = viz_content[pn]

    @get check_reactive(pn) = begin
        # Run all warmup strategies, clearing failed caches first
        for w in warmup_strategies()
            @is_cached(reactive_result[pn, w]) && !reactive_result[pn, w].ok && @clear_cache! reactive_result[pn, w]
            reactive_result[pn, w]
        end
        viz_content[pn]
    end

    @get fragment_table = overview

    viz_picker_content = begin
        items = []
        for name in posterior_names
            has_reactive = any(w -> @is_cached(reactive_result[name, w]), warmup_strategies())
            has_compile = @is_cached(compile_result[name])
            has_reactive || has_compile || continue
            n_strats = sum(w -> @is_cached(reactive_result[name, w]) && reactive_result[name, w].ok, warmup_strategies())
            badge = has_reactive ? "●" : "○"
            badge_color = has_reactive ? "green" : "#888"
            label = has_reactive ? "$name ($n_strats/$(length(warmup_strategies())))" : name
            push!(items, h.a(
                h.span(badge; style="color:$badge_color;margin-right:4px"),
                label;
                style="display:block;padding:6px 8px;text-decoration:none;border-radius:4px;margin-bottom:2px;cursor:pointer",
                hx_get="/fragment_viz/$name",
                hx_target="#content",
                hx_swap="innerHTML",
                hx_push_url="/viz/$name",
            ))
        end
        bc = breadcrumb([
            ("Table", "/fragment_table", "/"),
            ("Viz", nothing, nothing),
        ])
        isempty(items) && return h.div(
            bc,
            h.h3("Trajectory Visualization"),
            h.p("No posteriors have been sampled yet. Run sampling from the Table view first."),
        )
        h.div(
            bc,
            h.h3("Trajectory Visualization"),
            h.p("Select a posterior (● = has viz data, shows strategies cached):"; style="font-size:0.85em;color:#888;margin-bottom:8px"),
            items...,
        )
    end

    @get fragment_viz_picker = viz_picker_content

    viz_content(pn) = begin
        # Collect all cached results (ok and failed)
        available = Pair{String,Any}[]
        failed = Pair{String,Any}[]
        for w in warmup_strategies()
            if @is_cached(reactive_result[pn, w])
                r = reactive_result[pn, w]
                r.ok ? push!(available, w => r) : push!(failed, w => r)
            end
        end

        bc = breadcrumb([
            ("Table", "/fragment_table", "/"),
            ("Viz", "/fragment_viz_picker", "false"),
            (pn, nothing, nothing),
        ])

        if isempty(available) && isempty(failed)
            return h.div(
                bc,
                h.p("No reactive NUTS data for $pn yet."),
                h.a("Run reactive NUTS sampling"; href="/check_reactive/$pn"),
            )
        end

        # Default to first successful strategy
        default_w = isempty(available) ? nothing : first(available).first

        # Strategy selector table — successful rows
        rows = map(available) do (w, r)
            min_ess = round(r.min_ess; digits=1)
            med_ess = round(r.median_ess; digits=1)
            is_default = w == default_w
            h.tr(
                h.td(warmup_label(w)),
                h.td(h.span("PASS"; style="color:green;font-weight:bold")),
                h.td(string(med_ess)),
                h.td(string(min_ess)),
                h.td(string(r.n_divergent)),
                h.td(string(round(r.time; digits=2), "s")),
                h.td(h.button("Show"; style="padding:2px 8px;font-size:0.8em",
                    hx_get="/fragment_viz_single/$pn/$w",
                    hx_target="#viz-panel",
                    hx_swap="innerHTML",
                ));
                style=is_default ? "font-weight:bold" : "",
            )
        end

        # Failed rows with error info
        fail_rows = map(failed) do (w, r)
            h.tr(
                h.td(warmup_label(w)),
                h.td(h.span("FAIL"; style="color:red;font-weight:bold")),
                h.td(; colspan="4")(h.code(something(r.error, "unknown error"); style="font-size:0.8em")),
                h.td("");
                style="opacity:0.7",
            )
        end

        # Error details (collapsible stacktraces)
        error_details = map(failed) do (w, r)
            isnothing(r.stacktrace) && return ""
            h.details(; style="margin-bottom:4px;font-size:0.8em")(
                h.summary("$(warmup_label(w)) stacktrace"),
                h.pre(r.stacktrace; style="white-space:pre-wrap;max-height:200px;overflow:auto;font-size:0.85em"),
            )
        end

        selector = h.table(; role="grid", style="font-size:0.85em;margin-bottom:8px")(
            h.thead(h.tr(
                h.th("Strategy"), h.th("Status"), h.th("Med ESS"), h.th("Min ESS"),
                h.th("Div"), h.th("Time"), h.th(""),
            )),
            h.tbody(rows..., fail_rows...),
        )

        init = isnothing(default_w) ? "" : viz_init_script[pn, default_w]

        viz_panel = if isnothing(default_w)
            h.div(; id="viz-panel")(h.p("All strategies failed. See errors above."))
        else
            h.div(; id="viz-panel")(
                h.div(; id="shared-traces", style="max-width:900px;display:flex;gap:6px"),
                h.div(; style="max-width:900px")(
                    h.div(warmup_label(default_w); class="panel-label", id="panel-label"),
                    h.div(class="mcmc-grid")(
                        h.div(; class="panel viz", id="viz-1"),
                        h.div(; class="panel hist-y", id="hist-y-1"),
                        h.div(; class="panel trace-y", id="trace-y-1"),
                        h.div(; class="panel hist-x", id="hist-x-1"),
                        h.div(; class="panel trace-x", id="trace-x-1"),
                    ),
                ),
                h.div(; style="max-width:900px;margin:8px 0;display:flex;align-items:center;gap:10px")(
                    h.button("▶"; id="btn-play", style="min-width:36px;padding:6px 10px"),
                    h.canvas(; id="progress-bar", style="flex:1;height:20px"),
                    h.div(; style="display:flex;align-items:center;gap:6px")(
                        h.button("÷2"; id="btn-slow", style="min-width:36px;padding:6px 10px"),
                        h.span(; id="speed-display", class="speed-display"),
                        h.button("×2"; id="btn-fast", style="min-width:36px;padding:6px 10px"),
                    ),
                ),
                h.script(; src="/serve_static/mcmc-viz.js"),
                h.div(init; id="viz-init"),
            )
        end

        [
            h.style(viz_css()),
            bc,
            h.div(; style="max-width:900px")(
                h.h3("$pn"; id="viz-title", style="margin-bottom:4px"),
                selector,
                error_details...,
            ),
            viz_panel,
        ]
    end

    @get fragment_viz(pn) = viz_content[pn]

    @get fragment_viz_single(pn, w) = begin
        r = reactive_result[pn, w]
        init = viz_init_script[pn, w]
        [
            h.div(; id="shared-traces", style="max-width:900px;display:flex;gap:6px"),
            h.div(; style="max-width:900px")(
                h.div(warmup_label(w); class="panel-label", id="panel-label"),
                h.div(class="mcmc-grid")(
                    h.div(; class="panel viz", id="viz-1"),
                    h.div(; class="panel hist-y", id="hist-y-1"),
                    h.div(; class="panel trace-y", id="trace-y-1"),
                    h.div(; class="panel hist-x", id="hist-x-1"),
                    h.div(; class="panel trace-x", id="trace-x-1"),
                ),
            ),
            h.div(; style="max-width:900px;margin:8px 0;display:flex;align-items:center;gap:10px")(
                h.button("▶"; id="btn-play", style="min-width:36px;padding:6px 10px"),
                h.canvas(; id="progress-bar", style="flex:1;height:20px"),
                h.div(; style="display:flex;align-items:center;gap:6px")(
                    h.button("÷2"; id="btn-slow", style="min-width:36px;padding:6px 10px"),
                    h.span(; id="speed-display", class="speed-display"),
                    h.button("×2"; id="btn-fast", style="min-width:36px;padding:6px 10px"),
                ),
            ),
            h.script(; src="/serve_static/mcmc-viz.js"),
            h.div(init; id="viz-init"),
        ]
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
        try
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
        catch e
            push!(lines, "ERROR: $(sprint(showerror, e, catch_backtrace()))")
        end
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
            r = force_check[check, pn]
            push!(results, "$(r.ok ? "PASS" : "FAIL") $pn$(r.ok ? "" : " -- $(r.error)")")
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
                h.button("Pathfinder"; hx_get="/example_pathfinder", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
                h.button("WarmupHMC"; hx_get="/example_sampling", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
                h.button("CmdStan"; hx_get="/example_cmdstan", hx_include="#examples-select", hx_target="#examples-result", hx_swap="innerHTML"),
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
