@testitem "cooperative cluster geometry" setup=[MetricRestartFixtures] begin
    using LinearAlgebra, Random, Test, WarmupHMC
    const WH = WarmupHMC

    @testset "capacity-limited responsibility quotas" begin
        for count in (1024, 2048)
            q = WH._halo_quotas(ones(count), ones(Int, count), 1000)
            @test sum(q) == 1000 && all(x -> x in (0, 1), q)
        end
        @test WH._halo_quotas([0.75, 0.25], [100, 100], 20) == [15, 5]
        @test WH._halo_quotas([0.5, 0.5], [100, 1], 20) == [19, 1]
        rng = Xoshiro(9923)
        for _ in 1:100
            weights = rand(rng, 0:4, 20)
            capacity = rand(rng, 0:8, 20)
            budget = rand(rng, 0:60)
            q = WH._halo_quotas(weights, capacity, budget)
            @test sum(q) == min(budget, sum(capacity[weights .> 0]))
            @test all(0 .<= q .<= capacity) && all(iszero, q[weights .== 0])
        end
    end

    target, sq = corr_gaussian(1)
    function donor(p, g)
        d = size(p, 1)
        cfg = (; recording_target=1000, n_evaluations=1000, initializer_kw=(;),
               target_acceptance_rate=0.8, monitor_ess=false)
        chain = WH._coop_chain(1, Xoshiro(8), target, (; position=zeros(d), squared_scale=sq),
                               cfg; progress=nothing)
        append!(chain.recording_lpdf.halo_position, p)
        append!(chain.recording_lpdf.halo_gradient, g)
        for (x, grad) in zip(eachcol(p), eachcol(g))
            WH._fit_moments!(chain.moments, x, grad)
        end
        chain
    end
    p, g = gaussian_halo(target, 240)
    p0, g0 = copy(p), copy(g)
    d = size(p, 1)
    chains = [donor(p, g), donor(p, g)]
    cfg = (; max_reflections=2, recording_target=240, n_cores=2)
    clusters = WH.CoopClusters()
    WH._pool_clusters!(clusters, chains, [1.0 0.0; 0.0 1.0])
    geometry = WH._cluster_geometry(clusters, chains, cfg; eligible=[true, false])
    @test all(c -> c.recording_lpdf.halo_position == p0 && c.recording_lpdf.halo_gradient == g0, chains)
    selected = WH._cluster_metric(geometry, 1)
    @test selected.scale isa WH.ScaleThenReflect
    @test 0 < length(selected.scale.m1.idxs) <= 2
    @test WH._cluster_metric(geometry, 2).weight == 0
    @test WH._cluster_metric(geometry, 2).scale isa Diagonal
    rescued = WH._cluster_metric(geometry, 2, 1)
    chains[2].geometry_current = false
    @test isnothing(WH._cluster_geometry(clusters, chains, cfg; eligible=trues(2)).factors[2])
    chains[2].geometry_current = true
    @test rescued.scale === selected.scale && rescued.weight == selected.weight
    @test isnothing(WH._fit_cluster_factor!(p[:, 1:2], g[:, 1:2], 2))
    @test isnothing(WH._fit_cluster_factor!(copy(p), zeros(size(g)), 2))
    @test all(isnothing, WH._cluster_geometry(clusters, chains,
        merge(cfg, (; max_reflections=0)); eligible=trues(2)).factors)
    singleton, singleton_grad = WH._cluster_sample(chains, [1, 0], [240, 240], 1)
    @test singleton[:, 1] == p[:, 120] && singleton_grad[:, 1] == g[:, 120]

    @testset "each occupied cluster uses its own factor" begin
        other = donor(reverse(p; dims=1), reverse(g; dims=1))
        donors = [chains[1], other]
        separate = WH._pool_clusters!(WH.CoopClusters(), donors, [1.0 0.0; 0.0 1.0])
        one = WH._cluster_geometry(separate, donors, merge(cfg, (; n_cores=1)))
        two = WH._cluster_geometry(separate, donors, cfg)
        @test all(L -> L isa WH.ScaleThenReflect, one.factors)
        @test WH.scale_data.(one.factors) == WH.scale_data.(two.factors)
        @test WH.scale_data(one.factors[1]) != WH.scale_data(one.factors[2])
        @test all(i -> WH._cluster_metric(two, i).scale === two.factors[i], 1:2)
    end

    @testset "zero reflections retain the accumulated diagonal" begin
        # This halo already fits in its original coordinates. Other leaves
        # in this chain's history give a different pooled diagonal.
        raw = randn(Xoshiro(102), d, 240)
        c = donor(raw, -raw)
        for x in eachcol(raw)
            WH._fit_moments!(c.moments, 4x, -x)
        end
        cl = WH._pool_clusters!(WH.CoopClusters(), [c], ones(1, 1))
        geom = WH._cluster_geometry(cl, [c], cfg)
        @test isnothing(only(geom.factors))
        pooled = WH._cluster_metric(geom, 1).scale
        @test pooled isa Diagonal
        @test pooled.diag ≈ WH._moment_scales(c.moments)
        @test all(>(1.5), pooled.diag)
    end

    @testset "within-target fits ignore independent shifts in position and gradient" begin
        shifts = collect(1.0:d)
        shifted = [donor(p .+ shifts, g .- 2shifts), donor(p .- 3shifts, g .+ shifts)]
        weights = [0.5, 0.5]
        capacities = [240, 240]
        centered(cs) = WH._cluster_sample(cs, weights, capacities, 240; within_target=true)
        x0, y0 = centered(chains)
        x, y = centered(shifted)
        @test x ≈ x0 && y ≈ y0
        raw, raw_grad = WH._cluster_sample(shifted, weights, capacities, 240)
        @test raw[:, 1] == p[:, 1] + shifts && raw_grad[:, 1] == g[:, 1] - 2shifts
        function factor(cs)
            cl = WH._pool_clusters!(WH.CoopClusters(), cs, ones(2, 1))
            only(WH._cluster_geometry(cl, cs, cfg; eligible=trues(2), within_target=true).factors)
        end
        A, B = factor(chains), factor(shifted)
        @test A isa WH.ScaleThenReflect && B isa WH.ScaleThenReflect
        x = randn(Xoshiro(676), d)
        @test (A * (A' * x)) ≈ (B * (B' * x))
    end

    @testset "structured restart agrees with dense reference" begin
        L = selected.scale
        dense = L * Matrix{Float64}(I, d, d)
        @test WH._metric_variances(L) ≈ vec(sum(abs2, dense; dims=2))
        @test WH._complete_diagonal(fill(NaN, d), L).diag ≈ sqrt.(diag(dense * dense'))
        unsupported = WH._cluster_metric_choice(L, nothing, WH.CoopMoments(d))
        @test unsupported.scale === L && unsupported.relative_condition == 1
        @test selected.relative_condition ≈ cond(chains[1].scale \ dense) rtol=1e-5
        # The diagonal restart test ignores unsupported coordinates even when
        # every supported scale changes by the same factor.
        @test WH._relative_condition([NaN, 2.0, 2.0], Diagonal(ones(3))) == 1
    end
end

@testitem "cooperative reflection checkpoints and streams" setup=[MetricRestartFixtures] begin
    using LinearAlgebra, Random, Serialization, Test, WarmupHMC
    const WH = WarmupHMC
    target, sq = corr_gaussian(12; narrow=10)
    opts = (; n_chains=2, init=(; position=zeros(length(sq)), squared_scale=sq),
             n_draws=180, n_evaluations=500, max_reflections=3)
    sample(cores; kw...) = cooperative_warmup_mcmc(Xoshiro(52), target; n_cores=cores, opts..., kw...)
    serial = sample(1)
    parallel = sample(2)
    @test all(r -> r.scale isa WH.ScaleThenReflect, serial.results)
    @test !Base.mightalias(serial.results[1].scale.m2.diag, serial.results[2].scale.m2.diag)
    @test all(r -> r.scale isa Diagonal, sample(1; max_reflections=0).results)
    @test_throws ArgumentError sample(1; max_reflections=-1)
    function check_equivalent(out)
        @test serial.stop_reason === out.stop_reason === :n_draws
        for (a, b) in zip(serial.results, out.results)
            @test a.posterior_position == b.posterior_position
            @test a.stepsize == b.stepsize && a.total_evaluation_counter == b.total_evaluation_counter
            @test WH.scale_data(a.scale) == WH.scale_data(b.scale)
        end
    end
    check_equivalent(parallel)
    mktempdir() do dir
        sample(1; n_evaluations_budget=15_000, checkpoint_dir=dir)
        state = deserialize(joinpath(dir, "run_latest.jls"))
        @test any(p -> p.active_transformation === :adaptive, state.chains)
        for p in state.chains
            @test p.scale_options[p.active_transformation] isa WH.ScaleThenReflect
            step = fixed_kernel_nuts_step(FixedKernelNUTSState(p), target)
            @test all(isfinite, step.state.position)
        end
        rm(joinpath(dir, "run_summary.json"))  # crash after the saved round
        check_equivalent(sample(2; checkpoint_dir=dir, resume=true))
    end
end
