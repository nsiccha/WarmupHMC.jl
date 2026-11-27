@testitem "Continuous centering carries frame-dependent history, uses cached gradients, and replays" begin
    using WarmupHMC, Random, Enzyme, DifferentiationInterface, LogDensityProblems, Statistics, LinearAlgebra
    include(joinpath(@__DIR__, "checkpoint_cases.jl"))
    using .AdaptiveCheckpointCases: latest, stop_after
    const W = WarmupHMC
    const LDP = LogDensityProblems
    struct JointFunnel
        calls::Base.RefValue{Int}
        reject::Base.RefValue{Bool}
    end
    Base.deepcopy_internal(p::JointFunnel, ::IdDict) = p
    LDP.dimension(::JointFunnel) = 4
    LDP.capabilities(::Type{JointFunnel}) = LDP.LogDensityOrder{1}()
    function LDP.logdensity_and_gradient(p::JointFunnel, q)
        p.calls[] += 1
        p.reject[] && return (-Inf, fill(NaN, 4))
        v, x, y, spectator = q
        s = exp(-v)
        t = (x^2 + y^2)*s
        -v^2/2 - t/2 - v - spectator^2/18, [-v + t/2 - 1, -x*s, -y*s, -spectator/9]
    end
    make_rules() = IndexedReparametrization([i => Reparametrization(
        PartiallyCentered(1.), PartiallyCentered(0.45), 0., q -> q[1]/2) for i in 2:3])
    # The second cell's score also depends on the first cell's installed
    # source. First cell defers to the generic scalar map.
    plan(sync!) = CandidateScoringPlan((ir,q,g) -> (ir.pairs[1].second.source.c,q,g),
        (f,j,i,r,c) -> j == 1 ? nothing : (0., f[2][i] + c.c*f[1]*f[2][1],
                                                   f[3][i] + c.c*f[1]*f[3][1]); synchronize! = sync!)
    function problem(; reject_moves=false)
        model = JointFunnel(Ref(0), Ref(false))
        sync! = ir -> model.reject[] = reject_moves && any(p -> last(p).source.c != 0.45, ir.pairs)
        ReparametrizedProblem(make_rules(), model, AutoEnzyme(); scoring_plan=plan(sync!))
    end
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
        counter = only(result.completion.chains)
        @test counter.gradient_calls == live.problem.calls[]
        @test counter.gradient_calls == counter.initialization_gradient_calls + counter.total_evaluation_counter +
                                       length(r.centering_installations) + counter.n_warmup_windows
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
        # The second scorer reads the first child's installed control. Check its
        # offsets and GN matrix against a batch calculation in the new chart.
        c1, c2 = h.parameters[1,:]
        x = qnew[3,:] .+ c2*c1 .* qnew[1,:]
        g = gnew[3,:] .+ c2*c1 .* gnew[1,:]
        x̄, ḡ = mean(x), mean(g)
        @test h.parameters[2:3,2] ≈ [x̄, ḡ]
        k = exp(-h.parameters[4,2]/2)
        H = Matrix{Float64}(I,4,4)
        for i in eachindex(x)
            jx = [c1*qnew[1,i]*k, -k, 0., -(x[i]-x̄)*k/2]
            jg = [c1*gnew[1,i]/k, 0., -1/k, (g[i]-ḡ)/(2k)]
            H .+= jx*jx' + jg*jg'
        end
        @test h.information[:,:,2] ≈ H
        expected = sqrt(std(P[4,:]) / std(G[4,:]))
        @test cp0.scale_options.diagonal.diag[4] ≈ expected
        @test last(LDP.logdensity_and_gradient(newp, cp0.position_and_gradient.q)) ≈ cp0.position_and_gradient.∇ℓq
        @test all(j -> isposdef(Symmetric(h.information[:,:,j])), 1:2)
        transferred = r.centering_handoff
        @test transferred.frozen && transferred.stepsize == r.stepsize
        @test transferred.ensemble.source_indices == [2,3]
        next_problem = problem()
        W.restore_reparam_sources!(next_problem, transferred.reparam_sources)
        for (i,(q,g)) in enumerate(zip(eachcol(transferred.ensemble.positions), eachcol(transferred.ensemble.gradients)))
            @test last(W.reparametrizer(next_problem)(collect(q))) ≈ r.posterior_position[:,i]
            @test last(LDP.logdensity_and_gradient(next_problem, collect(q))) ≈ g
        end
        restarted = adaptive_warmup_mcmc(Xoshiro(27), next_problem; cfg..., init=transferred.init,
            centering_schedule=:continuous, centering_initial_ensemble=transferred.ensemble,
            stepsize_adaptation_limit=4, n_draws=2)
        @test restarted.initializer_rows == cfg.n_draws
        # Interrupt before the freeze and resume without re-supplying rows.
        adaptive_warmup_mcmc(Xoshiro(3), problem(); cfg..., centering_schedule=:continuous,
            centering_initial_ensemble=ensemble, checkpoint_dir=partial, stop_requested=stop_after(10))
        cp = latest(partial)
        @test 0 < cp.continuous_centering.trees < cfg.stepsize_adaptation_limit
        @test cp.stepsize_state.m == cp.current_transition_counter + 1
        decay = exp2(-1/W._DEFAULT_CONTINUOUS_HALF_LIFE)
        nt = cp.continuous_centering.trees
        @test cp.continuous_centering.moments.weight ≈ size(P,2)*decay^nt + sum(decay^i for i in 0:nt-1)
        @test sum(cp.continuous_centering.anchors.weights[1:cp.continuous_centering.anchors.seeds]) ≈ size(P,2)*decay^nt
        # Identical anchors, different historical sufficient statistics.
        # Shared-parent moments and the dependent cell must still distinguish
        # the two histories after an installation. The spectator is invariant.
        function transport_history(factor)
            oldp = problem()
            W.restore_reparam_sources!(oldp, cp.reparam_sources)
            s = W._continuous_leaves(oldp, true, :halo, deepcopy(cp.continuous_centering)).schedule
            s.moments.gradient_m2[1] *= factor
            info = copy(s.information)
            info[:,:,2] .*= factor
            next = copy(s.parameters)
            next[1,:] .= (0.73, 0.26)
            q, g, w = W._anchor_sample(s.anchors)
            previous = W._anchor_scores(next, oldp, q, g)
            newp = problem()
            W.restore_reparam_sources!(newp, [i=>PartiallyCentered(next[1,j]) for (j,(i,_)) in enumerate(cp.reparam_sources)])
            @test W._transport_joint_history!(s, next, info, newp, W.reparametrizer(oldp), previous, q, g, w)
            @test s.moments.gradient_m2[4] == cp.continuous_centering.moments.gradient_m2[4]
            s.moments.gradient_m2[1], info[:,:,2]
        end
        v1, H1 = transport_history(1.)
        v3, H3 = transport_history(3.)
        @test v3 ≈ 3v1
        @test H3 ≈ 3H1
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(8), problem(); cfg...,
            checkpoint_dir=partial, resume=true, nonlinear_evidence=:linear_pool)
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
        fixed = joinpath(root,"fixed")
        fixedkw = (; centering_schedule=:continuous, nonlinear_adapt=false,
                     nonlinear_trajectory_weighting=:stepsize, centering_initial_ensemble=ensemble)
        fixed_full = adaptive_warmup_mcmc(Xoshiro(11), problem(); cfg..., fixedkw...)
        adaptive_warmup_mcmc(Xoshiro(11), problem(); cfg..., fixedkw..., checkpoint_dir=fixed,
                             stop_requested=stop_after(10))
        cp = latest(fixed)
        fixed_nt = cp.continuous_centering.trees
        @test cp.continuous_centering.moments.weight > size(P,2)*decay^fixed_nt
        @test cp.continuous_centering.moments.weight != size(P,2)*decay^fixed_nt + sum(decay^i for i in 0:fixed_nt-1)
        fixed_resumed = adaptive_warmup_mcmc(Xoshiro(10), problem(); cfg..., checkpoint_dir=fixed, resume=true)
        @test isempty(fixed_resumed.centering_installations)
        @test fixed_resumed.posterior_position == fixed_full.posterior_position
        @test fixed_resumed.scale_options.diagonal.diag == fixed_full.scale_options.diagonal.diag
        rejecting = problem(; reject_moves=true)
        rejectdir = joinpath(root,"rejection")
        rejected = adaptive_warmup_mcmc(Xoshiro(31), rejecting; cfg..., centering_schedule=:continuous,
            centering_initial_ensemble=ensemble, checkpoint_dir=rejectdir, stop_requested=stop_after(4))
        cp = latest(rejectdir)
        @test isempty(rejected.centering_installations) && !rejecting.problem.reject[]
        @test cp.continuous_centering.parameters[1,:] == ensemble.source_controls
        @test cp.position_and_gradient.∇ℓq ≈ last(LDP.logdensity_and_gradient(rejecting, cp.position_and_gradient.q))
        approx = W._map_joint_marginal([1.,2.,3.], [-1.,-3.,-5.], ones(3), 12., 17., false)
        @test approx.mean ≈ -23. && approx.m2 ≈ 68.
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
