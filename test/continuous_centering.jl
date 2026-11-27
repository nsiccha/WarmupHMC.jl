@testitem "Continuous centering carries frame-dependent history, uses cached gradients, and replays" begin
    using WarmupHMC, Random, Enzyme, DifferentiationInterface, LogDensityProblems, Statistics, LinearAlgebra
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: latest, stop_after, comparable, same
    const W = WarmupHMC
    const LDP = LogDensityProblems
    struct JointFunnel
        calls::Base.RefValue{Int}
    end
    Base.deepcopy_internal(p::JointFunnel, ::IdDict) = p
    LDP.dimension(::JointFunnel) = 4
    LDP.capabilities(::Type{JointFunnel}) = LDP.LogDensityOrder{1}()
    function LDP.logdensity_and_gradient(p::JointFunnel, q)
        p.calls[] += 1
        v, x, y, spectator = q
        s = exp(-v)
        t = (x^2 + y^2)*s
        -v^2/2 - t/2 - v - spectator^2/18, [-v + t/2 - 1, -x*s, -y*s, -spectator/9]
    end
    make_rules() = IndexedReparametrization([i => Reparametrization(
        PartiallyCentered(1.), PartiallyCentered(0.45), 0., q -> q[1]/2) for i in 2:3])
    # The second cell's score also depends on the first cell's installed
    # source. First cell defers to the generic scalar map.
    plan() = CandidateScoringPlan((ir,q,g) -> (ir.pairs[1].second.source.c,q,g),
        (f,j,i,r,c) -> j == 1 ? nothing : (0., f[2][i] + c.c*f[1]*f[2][1],
                                                   f[3][i] + c.c*f[1]*f[3][1]))
    problem() = ReparametrizedProblem(make_rules(), JointFunnel(Ref(0)), AutoEnzyme(); scoring_plan=plan())
    p = problem()
    rng = Xoshiro(719)
    P = randn(rng,4,64)
    P[4,:] .*= 3
    G = reduce(hcat, [last(LDP.logdensity_and_gradient(p, collect(q))) for q in eachcol(P)])
    ensemble = (; positions=P, gradients=G, source_controls=[0.45,0.45])
    savedP, savedG = copy(P), copy(G)
    cfg = (; init=(; position=[0.3,-0.4,0.2,0.1], squared_scale=ones(4)),
            n_draws=18, stepsize_adaptation_limit=24, n_evaluations=30,
            max_tree_depth=5, max_divergent_fraction=1., recording_target=8)
    mktempdir() do root
        full, partial = joinpath(root,"full"), joinpath(root,"partial")
        live = problem()
        result = completion_warmup_mcmc([Xoshiro(3)], live; cfg..., centering_schedule=:continuous,
                    centering_initial_ensemble=ensemble, checkpoint_dir=full)
        r = only(result.results)
        @test r.initializer_rows == size(P,2)
        @test r.adaptation_trees == r.centering_frozen_after == 23
        @test !isempty(r.centering_installations)
        @test only(result.completion.chains).gradient_calls == live.problem.calls[]
        @test P == savedP && G == savedG
        # At initialization every cached row was available for exact transport.
        dir = joinpath(full,"chain_1")
        cp0 = W.deserialize(joinpath(dir,"cp_init.jls"))
        h = cp0.continuous_centering
        @test h.trees == 0 && h.supplied == size(P,2)
        @test h.parameters[1,:] != ensemble.source_controls
        @test all(0 .<= h.parameters[1,:] .<= 1)
        newp = problem()
        W.restore_reparam_sources!(newp, cp0.reparam_sources)
        qnew, gnew = similar(P), similar(G)
        W._jointly_transport_halo!(newp, W.reparametrizer(p), P, G, qnew, gnew)
        @test gnew[1,:] != G[1,:]  # gradient of the unchanged shared parent
        @test h.moments.gradient_m2[1] ≈ var(gnew[1,:]) * (size(P,2)-1)
        @test h.moments.gradient_m2[4] ≈ var(G[4,:]) * (size(P,2)-1)
        @test h.moments.position_m2[4] ≈ var(P[4,:]) * (size(P,2)-1)
        expected = ((sum(abs2, P[4,:] .- mean(P[4,:])) + 1) /
                    (sum(abs2, G[4,:] .- mean(G[4,:])) + 1))^0.25
        @test cp0.scale_options.diagonal.diag[4] ≈ expected
        @test last(LDP.logdensity_and_gradient(newp, cp0.position_and_gradient.q)) ≈ cp0.position_and_gradient.∇ℓq
        @test all(j -> isposdef(Symmetric(h.information[:,:,j])), 1:2)
        # Interrupt before the freeze and resume without re-supplying rows.
        adaptive_warmup_mcmc(Xoshiro(3), problem(); cfg..., centering_schedule=:continuous,
            centering_initial_ensemble=ensemble, checkpoint_dir=partial, stop_requested=stop_after(10))
        cp = latest(partial)
        @test 0 < cp.continuous_centering.trees < cfg.stepsize_adaptation_limit
        @test cp.stepsize_state.m == cp.current_transition_counter + 1
        resumed = adaptive_warmup_mcmc(Xoshiro(8), problem(); cfg..., checkpoint_dir=partial, resume=true)
        uninterrupted = adaptive_warmup_mcmc(Xoshiro(3), problem(); cfg..., centering_schedule=:continuous,
                                              centering_initial_ensemble=ensemble)
        @test resumed.posterior_position == uninterrupted.posterior_position
        @test resumed.stepsize == uninterrupted.stepsize
        @test resumed.scale_options.diagonal.diag == uninterrupted.scale_options.diagonal.diag
        @test resumed.centering_installations == uninterrupted.centering_installations
        @test resumed.initializer_rows == size(P,2)
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), problem(); cfg...,
            centering_schedule=:continuous, centering_initial_ensemble=merge(ensemble,(;source_controls=[0.,0.])),
            checkpoint_dir=partial, overwrite=true)
        @test latest(partial).total_transition_counter == cfg.stepsize_adaptation_limit + cfg.n_draws
    end
end

@testitem "Inactive continuous adaptation uses the diagonal algorithm for plain and fixed targets" begin
    using WarmupHMC, Random, LogDensityProblems
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: Banana, CONFIG
    cfg = merge(CONFIG,(;centering_schedule=:continuous, max_divergent_fraction=1.))
    plain(flag) = adaptive_warmup_mcmc(Xoshiro(450), Banana(); cfg..., nonlinear_adapt=flag)
    a, b = plain(true), plain(false)
    @test a.posterior_position == b.posterior_position
    @test a.stepsize == b.stepsize
    @test a.active_transformation == :diagonal
    @test a.scale_options.diagonal.diag != ones(2)
    @test isempty(a.centering_installations) && a.initializer_rows == 0
end
