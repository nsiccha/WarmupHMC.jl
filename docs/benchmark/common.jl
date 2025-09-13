using WarmupHMC, Plots, StanLogDensityProblems, DynamicHMC, Random, Distributions, PosteriorDB, LogDensityProblems, DataFrames, LinearAlgebra, JSON
using DynamicObjects, Term, Statistics, StatsBase
import Markdown
plotlyjs()
import DynamicHMC, QuartoComponents

import WarmupHMC: BayesianOptimizationStepsizeAdaptation, IPGPRegression, SquaredExponentialKernel, TransformedGPKernel, IntegratedSquaredExponentialKernel
import WarmupHMC: IPGPRegression, SquaredExponentialKernel, condition!, rescale!!, location!, qlocation!, qpred!
using Sobol
using LogExpFunctions
isfiniteorzero(x) = isfinite(x) ? x : zero(x)

randn_spd(rng, n) = begin 
    X = randn(rng, (n,n))
    X * X'
end

const project_dir = dirname(Base.active_project())
const pdb = PosteriorDB.database()
# @assert Threads.nthreads() > 1
stan_problem(path, data) = StanProblem(
    path, data;
    nan_on_error=true,
    make_args=["STAN_THREADS=TRUE"],
    warn=false
)
stan_problem(posterior_name::AbstractString) = stan_problem(
    PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, (posterior_name))), "stan")), 
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, (posterior_name))), String)
)
# stan_problem(n::Int) = stan_problem(
#     "docs/stan/std_normal.stan",
#     JSON.json((;n))
# )
stan_problem(scales::AbstractVector) = stan_problem(
    joinpath(project_dir, "stan/diag_normal.stan"),
    JSON.json((;n=length(scales), scales))
)
stan_problem(cov::AbstractMatrix) = stan_problem(
    joinpath(project_dir, "stan/full_normal.stan"),
    JSON.json((;n=size(cov, 1), cov=eachrow(cov)))
)

string_identifier(x::AbstractString) = x
string_identifier(x::AbstractVector) = "diag_normal($(length(x)), $(WarmupHMC.short_string(cond(Diagonal(x)))))"
string_identifier(x::AbstractMatrix) = "full_normal($(size(x, 1)), $(WarmupHMC.short_string(sqrt(cond(x)))))"
mappairs(f, x::NamedTuple) = map(f, keys(x), values(x))
mappairs2(f, x::NamedTuple) = (;zip(keys(x), mappairs(f, x))...)


mcmc(problem, position_and_gradient::DynamicHMC.EvaluatedLogDensity; rng, stepsize, n_draws=1000, max_depth=10, kwargs...) = begin
    dim = LogDensityProblems.dimension(problem) 
    draws = zeros((n_draws, 1, dim))
    n_steps = 0
    n_divergent = 0
    for i in 1:n_draws
        position_and_gradient, stats = WarmupHMC.sample_tree!(
            problem, position_and_gradient; rng, stepsize, max_depth, kwargs...
        )
        n_steps += stats.steps
        n_divergent += DynamicHMC.is_divergent(stats.termination)
        draws[i, 1, :] .= position_and_gradient.q
    end
    (;draws, n_steps, n_divergent)
end
da_adapt(
    problem, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    rng, stepsize, n_draws=1000, max_depth=10, target, kwargs...
) = begin
    stepsize_adaptation = DynamicHMC.DualAveraging(δ=target)
    stepsize_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, stepsize)
    cum_n_steps = zeros(Int, n_draws)
    tried_stepsizes = zeros(n_draws)
    selected_stepsizes = zeros(n_draws)
    for i in 1:n_draws
        stepsize = DynamicHMC.current_ϵ(stepsize_state)
        tried_stepsizes[i] = stepsize
        position_and_gradient, stats = WarmupHMC.sample_tree!(
            problem, position_and_gradient; rng, stepsize, max_depth, kwargs...
        )
        stepsize_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, stepsize_state, stats.acceptance_rate)
        cum_n_steps[i] = cum_n_steps[max(1, i-1)] + stats.steps
        selected_stepsizes[i] = DynamicHMC.final_ϵ(stepsize_state)
    end
    (;cum_n_steps, tried_stepsizes, selected_stepsizes)
end
metric_adapt(
    problem, position_and_gradient::DynamicHMC.EvaluatedLogDensity; 
    rng, stepsize, n_draws=1000, max_depth=10, kwargs...
) = begin 
    cum_n_steps = zeros(Int, n_draws)
    problem = WarmupHMC.NUTSPosterior(problem)
    dim = LogDensityProblems.dimension(problem)
    adaptations = (;
        stan=WarmupHMC.StanScaleAdaptation(dim), 
        nutpie=WarmupHMC.NutpieScaleAdaptation(dim), 
        intermediate_stan=WarmupHMC.IntermediateScaleAdaptation(dim; A=WarmupHMC.StanScaleAdaptation), 
        intermediate_nutpie=WarmupHMC.IntermediateScaleAdaptation(dim)
    )
    scales = map(x->zeros((dim, n_draws)), adaptations)
    for i in 1:n_draws
        WarmupHMC.reset!(problem)
        position_and_gradient, stats = WarmupHMC.sample_tree!(
            problem, position_and_gradient; rng, stepsize, max_depth, kwargs...
        )
        WarmupHMC.OnlineStatsBase.fit!(adaptations.stan, position_and_gradient.q)
        WarmupHMC.OnlineStatsBase.fit!(adaptations.nutpie, position_and_gradient.q, position_and_gradient.∇ℓq)
        WarmupHMC.OnlineStatsBase.fit!(adaptations.intermediate_stan, problem)
        WarmupHMC.OnlineStatsBase.fit!(adaptations.intermediate_nutpie, problem)
        map(adaptations, scales) do a, s
            WarmupHMC.marginal_scales!(Diagonal(view(s, :, i)), a)
        end
        cum_n_steps[i] = cum_n_steps[max(1, i-1)] + stats.steps
    end
    conds = map((;scales.stan, scales.nutpie)) do ref_scale
        map(scales) do scale
            cond.(Diagonal.(eachcol(Diagonal(ref_scale[:, end]) \ scale)))
        end
    end
    (;cum_n_steps, conds)
end
greedy_hdi(x; n) = begin 
    # n = ceil(Int, q * length(x))
    lo = hi = argmax(x)
    while hi - lo < n - 1
        if hi == length(x) 
            lo -= 1
        elseif lo == 1
            hi += 1
        elseif x[lo-1] > x[hi+1]
            lo -=1
        else
            hi += 1
        end
    end
    lo, hi
end
Plots.plot(gp::IPGPRegression; kwargs...) = plot!(plot(), gp; kwargs...)
Plots.plot!(p::Plots.Plot, gp::IPGPRegression; n=100, kwargs...) = Plots.plot!(p::Plots.Plot, range(extrema(gp.cache.x)..., n), gp::IPGPRegression; kwargs...)
Plots.plot!(p::Plots.Plot, x, gp::IPGPRegression; xlink=xlink(gp.link), ylink=ylink(gp.link), q=.1, fillalpha=.25, ylim=:auto, kwargs...) = begin 
    obs_y = ylink.(gp.cache.x, gp.cache.sum ./ gp.cache.n)
    ylim === :auto && plot!(p; ylim=collect(extrema(obs_y)))
    scatter!(p, xlink.(gp.cache.x), obs_y; kwargs...)
    vline!(p, xlink.(gp.inducing_x); linewidth=1, kwargs...)
    plot!(
        p, xlink.(x), ylink.(x, qpred!.(Ref(gp), x, 1-q));
        fillrange=ylink.(x, qpred!.(Ref(gp), x, q)),
        linewidth=0, fillalpha, kwargs...
    )
    plot!(
        p, xlink.(x), ylink.(x, qlocation!.(Ref(gp), x, 1-q));
        fillrange=ylink.(x, qlocation!.(Ref(gp), x, q)),
        linewidth=0, fillalpha, kwargs...
    )
    plot!(
        p, xlink.(x), ylink.(x, WarmupHMC.location!.(Ref(gp), x)); 
        kwargs...
    )
    p
end
no_link(x,y) = y
no_unlink(x, y) = y
eff_link(x,y) = y/exp(x)
eff_unlink(x, y) = y * exp(x)
xlink(::typeof(eff_link)) = exp
ylink(::typeof(eff_link)) = eff_unlink
eff_gp(log_stepsizes, effs; x_scale=.25) = begin
    gp = IPGPRegression(
        extrema(log_stepsizes)...; 
        kernel=SquaredExponentialKernel(log(2)*x_scale, 1.), 
        link=eff_link,
        # functions=(x->100,),
    )
    for (x, y) in zip(log_stepsizes, effs)
        isfinite(gp.link(x, y)) && condition!(gp, x, y)
    end
    rescale!!(gp; n=length(effs))
end
acc_link(x, y) = max(logitexp(y), -8)
acc_unlink(x, y) = logistic(y)
xlink(::typeof(acc_link)) = exp
ylink(::typeof(acc_link)) = acc_unlink
acc_gp(log_stepsizes, log_acceptance_rate; x_scale=.25) = begin
    gp = IPGPRegression(
        extrema(log_stepsizes)...; 
        kernel=SquaredExponentialKernel(log(2)*x_scale, 1.), 
        # kernel=TransformedGPKernel(
        #     (x, y)->exp(x)*y,
        #     IntegratedSquaredExponentialKernel(2, log(2)*x_scale, 1.),
        # ),
        link=acc_link,
        functions=(one, identity),
    )
    for (x, y) in zip(log_stepsizes, log_acceptance_rate)
        isfinite(gp.link(x, y)) && condition!(gp, x, y)
    end
    rescale!!(gp; n=length(log_acceptance_rate))
end

distance_from(x, vals) = minimum(val->abs(x-val), vals)
loss1(x) = pi .- distance_from(x, pi)
loss2(x) = 2/3*pi .- distance_from(x, (2/3*pi, 4/3*pi))
autoquarto(args...) = autoquarto(args)
autoquarto(x) = x
autoquarto(x::NamedTuple) = QuartoComponents.Tabset(map(autoquarto, x))
autoquarto(x::Union{Tuple,AbstractVector}) = QuartoComponents.Container(map(autoquarto, x))
autoquarto(x::AbstractString) = x

@dynamicstruct struct Experiment
    identifier
    v = v"0.0.9"
    sidentifier = string_identifier(identifier)
    jidentifier = isa(identifier, AbstractString) ? "\"$identifier\"" : identifier
    cache_path = joinpath(project_dir, "benchmark", "cache", sidentifier)
    problem = stan_problem(identifier)
    @cached dim = LogDensityProblems.dimension(problem)
    progress = nothing
    ess_kinds = (mean, :bulk, :tail, std, median, mad)
    @cached adaptive[seed] = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem; progress)
    scale[seed] = adaptive[seed].scale_options[adaptive[seed].active_transformation]
    @cached da[seed, target, reseed, n_draws] = da_adapt(
        problem, adaptive[seed].position_and_gradient;
        target, n_draws, rng=Xoshiro(reseed), stepsize=adaptive[seed].stepsize, scale=scale[seed], 
    )
    @cached ma[seed, reseed, n_draws] = metric_adapt(
        problem, adaptive[seed].position_and_gradient;
        n_draws, rng=Xoshiro(reseed), stepsize=adaptive[seed].stepsize, scale=scale[seed], 
    )
    # @cached uda[seed, target, reseed, n_draws] = da_adapt(
    #     problem, adaptive[seed].position_and_gradient;
    #     target, n_draws, rng=Xoshiro(reseed), stepsize=adaptive[seed].stepsize * lmin(scale[seed]), 
    # )
    # @cached da_stepsize[seed, target, reseed, n_draws] = da_adapt_stepsize(
    #     problem, adaptive[seed].position_and_gradient; 
    #     rng=Xoshiro(reseed), stepsize=adaptive[seed].stepsize, scale=adaptive[seed].scale_options[adaptive[seed].active_transformation], n_draws, target
    # )
    # @cached bo_stepsize[seed, reseed, n_draws] = bo_adapt_stepsize(
    #     problem, adaptive[seed].position_and_gradient; 
    #     rng=Xoshiro(reseed), stepsize=adaptive[seed].stepsize, scale=adaptive[seed].scale_options[adaptive[seed].active_transformation], n_draws
    # )
    @cached oracle_fit[seed, stepsize, n_draws] = mcmc(
        problem, adaptive[seed].position_and_gradient; 
        rng=Xoshiro(seed), stepsize, scale=scale[seed], n_draws
    )
    @cached oracle_effs[seed, stepsize, n_draws, kind] = begin
        (;draws, n_steps) = oracle_fit[seed, stepsize, n_draws]
        minimum(WarmupHMC.MCMCDiagnosticTools.ess(draws; kind)) / n_steps
    end
    oracle_eff[seed, stepsize, n_draws] = minimum(ess_kinds) do kind 
        oracle_effs[seed, stepsize, n_draws, kind]
    end
    oracle_gp_x[seed, i] = begin 
        lo, hi = log.(adaptive[seed].stepsize .* (.25, 2.))
        log_stepsizes = range(lo, hi, 2^(5+i)+1)
        log_stepsizes = vcat(log_stepsizes, range(hi, hi+log(2); step=(hi-lo)/2^(5+i))[2:end])
        lo, hi = extrema(log_stepsizes)
        stepsizes = exp.(log_stepsizes)
        if i > 0
            gp = deepcopy(o.oracle_gp[seed, i-1])
            gp_y = WarmupHMC.qlocation!.(Ref(gp), log_stepsizes, .9) .* stepsizes
            lo_idx, hi_idx = greedy_hdi(gp_y; n=length(o.oracle_gp_x[seed, 0]) * ceil(Int, 2. ^ (i/2)))
            log_stepsizes[lo_idx:hi_idx]
        else
            log_stepsizes
        end
    end
    oracle_gp[seed, i] = begin
        n_draws = 100 * 2^i
        log_stepsizes = oracle_gp_x[seed, i]
        lo, hi = extrema(log_stepsizes)
        stepsizes = exp.(log_stepsizes)
        if i > 0
            gp = deepcopy(o.oracle_gp[seed, i-1])
        else
            gp = IPGPRegression(lo, hi; kernel=SquaredExponentialKernel(log(2)/4, 1.), link=eff_link, functions=(x->100,))
        end
        effs = asyncmap(stepsizes) do stepsize
            1e3 * isfiniteorzero(oracle_eff[1, stepsize, n_draws])
        end
        for (x, y) in zip(log_stepsizes, effs)
            condition!(gp, x, fill(y, 2^i))
        end
        rescale!!(gp; n=length(effs))
    end
    oracle_gps[seed, i, kind] = begin
        n_draws = 100 * 2^i
        log_stepsizes = oracle_gp_x[seed, i]
        lo, hi = extrema(log_stepsizes)
        stepsizes = exp.(log_stepsizes)
        if i > 0
            gp = deepcopy(o.oracle_gps[seed, i-1, kind])
        else
            gp = IPGPRegression(lo, hi; kernel=SquaredExponentialKernel(log(2)/4, 1.), link=eff_link, functions=(x->100,))
        end
        effs = asyncmap(stepsizes) do stepsize
            1e3 * isfiniteorzero(oracle_effs[1, stepsize, n_draws, kind])
        end
        for (x, y) in zip(log_stepsizes, effs)
            condition!(gp, x, fill(y, 2^i))
        end
        rescale!!(gp; n=length(effs))
    end
    @cached stat[seed, i, stepsize] = begin 
        draw = collect(view(adaptive[seed].posterior_position, :, i))
        p = WarmupHMC.NUTSPosterior(problem)
        position_and_gradient, stats = WarmupHMC.sample_tree!(
            p, draw; rng=Xoshiro(i), stepsize, scale=scale[seed]
        )
        weights = WarmupHMC.leaf_weights!(zeros(1), p, stats.termination; cache=zeros(1))
        idxs = WarmupHMC.idxs(p)
        idx = findfirst(==(position_and_gradient.q), eachcol(p.position))
        # total_angle = (idxs[1] < 0 ? p.dangle[1] : 0) + (idxs[stats.steps] > 0 ? p.dangle[end] : 0) 
        (;
            stepsize, 
            n_steps=stats.steps, 
            log_acceptance_rate=logsumexp(min.(p.dH, 0)) - log(stats.steps),
            max_abs_dH=maximum(abs, p.dH),
            expected_angle=dot(weights, p.dangle), 
            expected_angle1=WarmupHMC.@bsum(weights * loss1(p.dangle)),
            expected_angle2=WarmupHMC.@bsum(weights * loss2(p.dangle)),
            expected_jump=WarmupHMC.@bsum(weights * stepsize * abs(idxs)),
            # expected_jump1=WarmupHMC.@bsum(weights * loss1(total_angle * abs(idxs) / stats.steps)),
            # expected_jump2=WarmupHMC.@bsum(weights * loss2(total_angle * abs(idxs) / stats.steps)),
            std_angle=std(p.dangle, Weights(weights)),
            std_jump=std(stepsize .* abs.(idxs), Weights(weights)),
            observed_angle=isnothing(idx) ? 0. : p.dangle[idx],
            observed_jump=isnothing(idx) ? 0. : stepsize * abs(idxs[idx])
        )
    end
    stat_scan[seed] = begin 
        lo, hi = log.(adaptive[seed].stepsize .* (.25, 4))#*sqrt(2)))
        draws = adaptive[seed].posterior_position
        s = SobolSeq(lo, hi)
        log_stepsizes = reduce(vcat, next!(s) for i in axes(draws, 2))
        stepsizes = exp.(log_stepsizes)
        asyncmap(axes(draws, 2)) do i
            stat[seed, i, stepsizes[i]]
        end |> DataFrame
    end

    da_plot[seed, target, reseed, n_draws] = begin
        (;cum_n_steps, tried_stepsizes, selected_stepsizes) = da[seed, target, reseed, n_draws]
        plot(
            cum_n_steps, 
            [tried_stepsizes, selected_stepsizes];
            xscale=:log10, yscale=:log10,
            label=["tried stepsize" "sampling stepsize"],
            xlabel="cum. no. leapfrog steps", ylabel="stepsize"
        )
    end
    ma_plot[seed, reseed, n_draws] = begin 
        (;cum_n_steps, conds) = ma[seed, reseed, n_draws]
        (;stan, nutpie) = conds
        ps = mappairs(conds) do key, value
            p = hline(
                [1, sqrt(2)]; 
                color=:black, label="", xscale=:log10, yscale=:log10, 
                xlabel="transition (log scale)", ylabel="condition number relative to baseline",
                title="baseline: $key estimate at $n_draws draws"
            )
            xticks = mappairs(value) do key, scales 
                plot!(p, value[key]; label=string(key))
                findfirst(<=(sqrt(2)), value[key])
            end
            xticks = filter(!isnothing, xticks)
            yticks = vcat(1, sqrt(2), getindex.(values(value), minimum(xticks))...)
            xticks = vcat(1, xticks...)
            ylim = [1, max(value[4][1], yticks[4])*1.1]
            plot!(p; 
                xticks=(xticks, xticks), xlim=[1, maximum(xticks)*1.1], 
                yticks=(yticks, vcat(1, "sqrt(2)", WarmupHMC.short_string.(yticks[3:end]))), ylim
            )
        end
        autoquarto((;nutpie=ps[2], stan=ps[1]))
    end
    stat_plot[seed] = begin 
        df = stat_scan[seed]
        df.expected_angle_eff = df.expected_angle ./ df.n_steps
        # df.expected_angle1_eff = df.expected_angle1 ./ df.n_steps
        # df.expected_angle2_eff = df.expected_angle2 ./ df.n_steps
        df.expected_jump_eff = df.expected_jump ./ df.n_steps
        effs = (;
            df.expected_angle_eff, #df.expected_angle1_eff, df.expected_angle2_eff, 
            df.expected_jump_eff, 
        )
        log_stepsizes = log.(df.stepsize)
        eff_gps = map(Base.Fix1(eff_gp, log_stepsizes), effs)
        reff_gps = (;
            min_eff=oracle_gp[seed, 3],
            mean_eff=oracle_gps[seed, 3, mean],
            std_eff=oracle_gps[seed, 3, std]
        )
        gps = merge(
            eff_gps,
            reff_gps,
            (;acceptance_rate=acc_gp(log.(df.stepsize), df.log_acceptance_rate))
        )
        eff_gps = merge(eff_gps, (;acceptance_rate=gps.acceptance_rate))
        elog_stepsizes = range(extrema(gps.min_eff.cache.x)..., 100)
        (;
            per_stepsize=[
                """
                All of the below plots visualize the (approximate) GP regression (in the "original" space), 
                and the raw measurements that went into the GP regression.

                The measurements are visualized as dots, while the GPs are visualized as follows:   
                
                * vertical lines: inducing point locations,
                * wiggly line: median of the estimate,
                * shaded areas:
                    * inner: 90% CrI of the estimate,
                    * outer: 90% CrI of new predictions.
                """,
                plot(
                    mappairs((key, gp)->plot(gp; color=1, title="$key", label="", markeralpha=.1), gps)...,
                    plot(gps.acceptance_rate; color=1, title="max(-8, logit(acceptance_rate))", label="", markeralpha=.1, ylink=no_unlink, xlabel="stepsize");
                    xscale=:log10, layout=(:, 1), size=(800, 1200), link=:x
                )
            ],
            per_efficiency=[
                """
                The three tabs visualize the (inferred via the GP regression) relationships between 
                the different measures of sampling efficiency (`min_eff`, `mean_eff`, and `std_eff`) and
                the different proxies (`expected_angle_eff` and `expected_jump_eff`) of trajectory statistics (`acceptance_rate`),
                where the proxies/potential optimization targets are normalized to obtain a maximal value of one to make them more easily comparable.

                Interpretation of the below plots should be as follows:

                * Ideally, there would be perfect correlation between the measures of sampling efficiency and and potential optimization targets - naturally, this is impossible.
                * The "proper" measure of adequacy of the optimization target is how close the measures of sampling efficiency are to their optimum if the optimization target reaches its optimum:
                    * the maximum of the measures sampling efficiency is obtained at the right border of the plot, and 
                    * the maximum of the potential optimization targets is obtained at the top border of the plot. 

                It goes without saying that for Dual Averaging (`acceptance_rate`), the optimization target would not be to maximize the acceptance rate,
                but to reach a pre-specified target.
                """,
                mappairs2(reff_gps) do key, ref
                    plot(
                        eff_unlink.(elog_stepsizes, WarmupHMC.location!.(Ref(ref), elog_stepsizes)),
                        map(collect(values(eff_gps))) do gp
                            normalize!(ylink(gp.link).(elog_stepsizes, WarmupHMC.location!.(Ref(gp), elog_stepsizes)), Inf)
                        end;
                        xlabel="Sampling efficiency ($key)", ylabel="(Bayesian) Optimization target",
                        label=permutedims(collect(string.(keys(eff_gps)))),
                        legend=:bottomright
                    )
                end,
            ]
        )
    end
    qc[seed, reseed] = autoquarto("""
        The `metric` and `stepsize` tabs visualize key quantities and relations that we exploit in making warm-up more efficient.

        All plots use the warm-up (i.e. step size and scale) and draws of a single chain, warmed-up using WarmupHMC.jl.
        Further chains/seeds may follow, but are currently not a priority.
        """,
        (;
            metric=[
                """
                The `running_condition` tab visualizes the running condition number of online estimates of the marginal scales of the posteriors,
                relative to 
                
                * different baselines and
                * different ways of estimating the marginal scales in an online way.

                Generally, the pair `stan`/`intermediate_stan` and the pair `nutpie`/`intermediate_nutpie` converge towards different target scales between pairs
                but the same target scales within pairs.
                Furthermore, `intermediate_*` style methods generally converge faster than their counterparts, 
                and `nutpie_*` style methods converge faster than `stan_` style methods.
                """,
                (;
                    running_condition=[
                        """
                        The `nutpie` and `stan` sub-tabs respectively use Stan or Nutpie style estimates of the marginal scales of the 
                        posterior using 1000 post-warm-up draws as the baseline. 

                        The different lines represent different ways of estimating the marginal scales, with

                        * `stan` using the positions of the draws,
                        * `nutpie` using the positions and gradients of the draws,
                        * `intermediate_stan` using the positions of appropriately selected intermediate NUTS states, and
                        * `intermediate_nutpie` using the positions and gradients of appropriately selected intermediate NUTS states.

                        The x-ticks represent the number of the transition at which the respective methods cross the threshold of `sqrt(2)`, and
                        the y-ticks represent the running condition number of all methods at the first time any of the methods crosses that threshold.

                        All methods use the same sequence of draws/trajectories. The baselines get computed from the same draws, 
                        and thus any estimates of convergence for the `stan` and `nutpie` methods will be overoptimistic for high transition numbers.
                        """,
                        ma_plot[seed, reseed, 1000]
                    ]
                )
            ],
            step_size=[
                """
                The `statistics` tab visualizes various step size dependent sampling statistics, 
                including "sampling run" specific ones:

                * `min_eff`: minimum over parameters and functions $ess_kinds sampling efficiency,
                * `mean_eff`: minimum over parameters sampling efficiency of the mean,
                * `std_eff`: minimum over parameters sampling efficiency of the standard deviation,

                but also "trajectory" specific ones, which can be used as Bayesian Optimization/Dual Averaging targets:

                * `expected_angle_eff`: a proxy of sampling efficiency computed from the appropriately weighted intermediate angles per NUTS trajectory,
                * `expected_jump_eff`: a proxy of sampling efficiency computed from the appropriately weighted intermediate integration times per NUTS trajectory,
                * `acceptance_rate`: `mean(min.(0, hamiltonian_errors))`per NUTS trajectory.

                All sampling run specific statistics get estimated per sampling run, 
                while all trajectory specific ones are estimated via a single NUTS trajectory
                starting from one of the obtained posterior draws.
                """,
                (;
                    # gp_regression=gp_plot[seed, 3],
                    statistics=stat_plot[seed],
                    # dual_averaging=da_plot[seed, .6, reseed, 100], 
                )
            ],
        )
    )

    qmd_path = joinpath(relpath(project_dir), "benchmark", "p", "$sidentifier.qmd")
    html_path = replace(qmd_path, ".qmd"=>".html")
    link = begin 
        mkpath(dirname(qmd_path))
        (isfile(qmd_path) && contains(read(qmd_path, String), "$v")) || open(qmd_path, "w") do fd 
            write(fd, """
            ---
            title: "Linear warm-up: $sidentifier (dim: $dim)"
            ---
            ```{julia}
            @debug "$v"
            include("../common.jl") 
            Experiment($jidentifier; cache_type=:parallel).qc[1, 1]
            ```
            """)
        end
        "[`$sidentifier`]($html_path)"
    end
    df_row = (;
        dim,
        link, 
        # mass_adaptation=@cache_status(o.ma[1, 1, 1000]), 
        # sampling_stats=@cache_status(o.stat_scan[1]),
        # sampling_efficiency=@cache_status(o.oracle_gp[1, 3]), 
        # dual_averaging=@cache_status(o.da[1, .6, 1, 100]), 
    )
end