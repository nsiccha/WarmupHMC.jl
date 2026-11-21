@testitem "Cooperative first draws from a slowly moving Gaussian pilot" setup=[WarmupHMCSharedFixtures] begin
    using LogDensityProblems, Random, Statistics, Serialization, WarmupHMC
    W = WarmupHMC
    targets = fill(DiagGaussian(zeros(16), ones(16)), 16)
    features = collect(1.:16.)
    slot = last(findall(iszero, W.CoopFamily(targets, nothing, features, 8).source))
    init = [(; position=zeros(16), squared_scale=ones(16)) for _ in targets]
    init[slot].position[end] = 30
    init[slot].squared_scale[end] = 1e-4
    for seed in 1:3
        out = cooperative_warmup_mcmc(Xoshiro(seed), targets; init, features, n_wave1=8, n_draws=1)
        @test all(c -> c.n_samples == 1 && c.status === :done, out.results)
        @test abs(out.results[slot].posterior_position[end,1]) < 5
        @test out.total_evaluation_counter == sum(c.total_evaluation_counter for c in out.results)
    end

    # Interrupt an independent same-target comparison and continue with its own RNG.
    cfg = (; family=false, n_draws=1, n_evaluations=1000, initializer_kw=(;),
        target_acceptance_rate=0.8, recording_target=100, monitor_ess=false,
        algorithm=W.DynamicHMC.NUTS(), stepsize_adaptation_limit=5, kappa_decay=0.9,
        kappa_prior_weight=2.)
    c = W._coop_chain(1, Xoshiro(19), targets[1], init[1], cfg; progress=nothing)
    total = Threads.Atomic{Int}(0)
    c.status = :probing
    append!(c.freeze.history, fill(c.position_and_gradient.ℓq - 8, 12))
    c.freeze.anchor = copy(c.position_and_gradient.q)
    W._coop_transition!(c, cfg, total)
    run = W.CoopRun(Xoshiro(42), targets[1], cfg, init[1], nothing, "resume", time_ns(),
        0., nothing, W.CoopChain[c], W.CoopClusters(), total, 1, 1, nothing, NaN)
    payload = deepcopy(W._coop_payload(c, run))
    @test payload.schema_version == 4 && c.n_samples == 0
    @test_throws ArgumentError FixedKernelNUTSState(payload)
    restored = W._restore_coop_chain(payload, deepcopy(c.lpdf), cfg; progress=nothing,
        adapt_centering=false, geometry_current=true)
    for (ch, counter) in ((c,total), (restored,Threads.Atomic{Int}(total[])))
        while ch.status === :probing
            W._coop_transition!(ch, cfg, counter)
        end
        @test ch.status === :sampling && ch.n_samples == 0
        W._coop_transition!(ch, cfg, counter)
    end
    @test W.chain_draws(c) == W.chain_draws(restored)
    @test c.freeze.probe_evaluations == restored.freeze.probe_evaluations > 0
end
