@testitem "ActiveReparametrizationState" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Pkg, TOML
    using Distributions
    using DifferentiationInterface, Enzyme

# The selected centering may change, but a warmup restart must retain the same
# physical chain state. Large locations make stale-source reuse unmistakable.
@testset "nonlinear restart preserves the active physical point" begin
    for evidence in (:linear_pool, :all_good_leaves), initial_c in (0.0, 1.0)
        @testset "$evidence from c=$initial_c" begin
            rng=Xoshiro(913)
            inner=DiagGaussian([0.5,500.0,-200.0],[0.5,1.0,2.0])
            ir=WarmupHMC.IndexedReparametrization([
                k=>WarmupHMC.Reparametrization(
                    WarmupHMC.PartiallyCentered(1.0),WarmupHMC.PartiallyCentered(initial_c),
                    inner.mu[k],x->x[1]) for k in 2:3
            ])
            rp=WarmupHMC.ReparametrizedProblem(ir,inner,AutoEnzyme())
            model=inner.mu .+ inner.sigma.*randn(rng,3,128)
            positions=reduce(hcat,(last(WarmupHMC._inverse_with_logabsdet_jacobian(ir,q))
                                  for q in eachcol(model)))
            gradients=reduce(hcat,(last(LogDensityProblems.logdensity_and_gradient(rp,q))
                                  for q in eachcol(positions)))
            old_point=WarmupHMC.DynamicHMC.evaluate_ℓ(rp,copy(positions[:,end]))
            old_source=copy(old_point.q)
            if evidence===:linear_pool
                point=WarmupHMC.find_reparametrization!(rp,positions,gradients,old_point)
            else
                recorder=WarmupHMC.NonlinearRecorder(rp;mode=evidence)
                for (q,g) in zip(eachcol(positions),eachcol(gradients))
                    WarmupHMC.OnlineStatsBase.fit!(ir,recorder.online,q,g)
                end
                point=WarmupHMC.find_reparametrization!(rp,recorder,positions,gradients,old_point)
            end
            @test all(last(p).c==1.0 for p in WarmupHMC.reparam_sources(rp))
            @test last(ir(point.q)) ≈ model[:,end] atol=1e-10
            @test point.q ≈ positions[:,end] atol=1e-10
            @test old_point.q==old_source
            value,gradient=LogDensityProblems.logdensity_and_gradient(rp,point.q)
            @test point.ℓq ≈ value
            @test point.∇ℓq ≈ gradient
            for j in 1:8:128
                @test last(ir(positions[:,j])) ≈ model[:,j] atol=1e-10
                @test gradients[:,j] ≈ last(LogDensityProblems.logdensity_and_gradient(rp,positions[:,j])) atol=1e-8
            end
        end
    end
end

@testset "unrepresentable centering updates preserve the complete old state" begin
    inner=DiagGaussian(zeros(2),ones(2))
    old_ir=WarmupHMC.IndexedReparametrization([
        2=>WarmupHMC.Reparametrization(WarmupHMC.PartiallyCentered(0.0),
            WarmupHMC.PartiallyCentered(0.0),0.0,x->x[1])])
    rp=WarmupHMC.ReparametrizedProblem(WarmupHMC.IndexedReparametrization(copy(old_ir.pairs)),inner,AutoEnzyme())
    old_position=[-1000.0 0.0; 1.0 0.5]
    old_gradient=reduce(hcat,(last(LogDensityProblems.logdensity_and_gradient(rp,q)) for q in eachcol(old_position)))
    active=WarmupHMC.DynamicHMC.evaluate_ℓ(rp,copy(old_position[:,end]))
    WarmupHMC.restore_reparam_sources!(rp,[2=>WarmupHMC.PartiallyCentered(1.0)])
    position=copy(old_position);gradient=copy(old_gradient)
    WarmupHMC._jointly_transport_halo!(rp,old_ir,old_position,old_gradient,position,gradient)
    @test !all(isfinite,gradient) # The existing transport cannot represent exp(-1000).
    logs,point=Test.collect_test_logs() do
        WarmupHMC._validated_transport!(rp,old_ir,old_position,old_gradient,position,gradient,active)
    end
    log=only(logs)
    @test log.level==Base.CoreLogging.Warn
    @test occursin(r"^Rejected centering update: moving the retained pool .* gave nonfinite", log.message)
    @test log.kwargs[:pool_states]==2
    @test log.kwargs[:nonfinite_pool_positions]==0
    @test log.kwargs[:nonfinite_pool_gradients]==1
    @test 2 in log.kwargs[:nonfinite_coordinates]
    @test (2=>(location=0.0,log_scale=-1000.0)) in log.kwargs[:endpoints]
    @test point === active
    @test position == old_position
    @test gradient == old_gradient
    @test WarmupHMC.reparam_sources(rp) == [2=>WarmupHMC.PartiallyCentered(0.0)]
    @test last(WarmupHMC.reparametrizer(rp)(point.q)) == old_position[:,end]
    value,g=LogDensityProblems.logdensity_and_gradient(rp,point.q)
    @test point.ℓq == value
    @test point.∇ℓq == g
end

# Snag completion-warmu-b757327c: a chain's current point whose new coordinates
# are not finite was rejected with "bad_positions = 0, bad_gradients = 0", the
# counts of a pool that had moved without a fault. The warning now names what was
# not finite, and a rejected update leaves the run as if the rule had kept the
# previous controls.

"""
    Poisoned(inner, poison)

`inner`, except that its next evaluation after `poison[] = :gradient` (or
`:log_density`) returns a `NaN` gradient (or log density), and `poison[]` reads
`:none` again. The transport of the pool never evaluates the target, so this
reaches exactly the evaluation of the chain's current point in new coordinates.
"""
struct Poisoned{P}
    inner::P
    poison::Base.RefValue{Symbol}
end
LogDensityProblems.capabilities(::Type{<:Poisoned}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Poisoned) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity(p::Poisoned, x) = LogDensityProblems.logdensity(p.inner, x)
function LogDensityProblems.logdensity_and_gradient(p::Poisoned, x)
    value, gradient = LogDensityProblems.logdensity_and_gradient(p.inner, x)
    poison, p.poison[] = p.poison[], :none
    poison === :gradient && return (value, fill(NaN, length(gradient)))
    poison === :log_density && return (NaN, gradient)
    (value, gradient)
end

@testset "a rejected update names the nonfinite quantity of the current point" begin
    poison=Ref(:none)
    inner=Poisoned(DiagGaussian(zeros(2),ones(2)),poison)
    pair(c)=[2=>WarmupHMC.Reparametrization(WarmupHMC.PartiallyCentered(0.0),
        WarmupHMC.PartiallyCentered(c),0.0,x->x[1])]
    old_ir=WarmupHMC.IndexedReparametrization(pair(0.0))
    pool=[0.1 -0.2 0.3; 0.5 -0.4 0.2]   # small log-scales: every state moves
    grad=reduce(hcat,(last(LogDensityProblems.logdensity_and_gradient(inner,q)) for q in eachcol(pool)))
    # Each case: the current point, the new centering, what poisons its evaluation,
    # the warning's reason, and its fields.
    cases=[
        # Log-scale 1000 at the current point: exp(1000) overflows its new source
        # coordinate, while the identity pair never evaluated it (WarmupHMC 0.2.21).
        ([1000.0,0.5],1.0,:none,r"current point has nonfinite coordinates in the new frame",
         f->f[:nonfinite_coordinates]==(2,) && f[:n_nonfinite_coordinates]==1 &&
            f[:endpoints]==(2=>(location=0.0,log_scale=1000.0),)),
        ([0.4,0.5],0.5,:gradient,r"gradient at the chain's current point is not finite",
         f->f[:nonfinite_coordinates]==(1,2) &&
            f[:endpoints]==(2=>(location=0.0,log_scale=0.4),)),
        ([0.4,0.5],0.5,:log_density,r"log density at the chain's current point is not finite",
         f->isnan(f[:log_density]) && collect(keys(f))==[:log_density]),
    ]
    for (q,c,poisoned,reason,fields) in cases
        rp=WarmupHMC.ReparametrizedProblem(WarmupHMC.IndexedReparametrization(pair(0.0)),inner,AutoEnzyme())
        active=WarmupHMC.DynamicHMC.evaluate_ℓ(rp,copy(q))
        @test isfinite(active.ℓq)
        WarmupHMC.restore_reparam_sources!(rp,[2=>WarmupHMC.PartiallyCentered(c)])
        position=copy(pool);gradient=copy(grad)
        poison[]=poisoned
        logs,point=Test.collect_test_logs() do
            WarmupHMC._validated_transport!(rp,old_ir,pool,grad,position,gradient,active)
        end
        @test poison[]===:none
        log=only(logs)
        @test log.level==Base.CoreLogging.Warn
        @test occursin(r"^Rejected centering update: ",log.message)
        @test occursin(reason,log.message)
        @test fields(Dict(log.kwargs))
        @test point===active
        @test position==pool && gradient==grad
        @test WarmupHMC.reparam_sources(rp)==[2=>WarmupHMC.PartiallyCentered(0.0)]
    end
    # Positive control: the same update without a fault at the current point moves it.
    rp=WarmupHMC.ReparametrizedProblem(WarmupHMC.IndexedReparametrization(pair(0.0)),inner,AutoEnzyme())
    active=WarmupHMC.DynamicHMC.evaluate_ℓ(rp,[0.4,0.5])
    WarmupHMC.restore_reparam_sources!(rp,[2=>WarmupHMC.PartiallyCentered(0.5)])
    position=copy(pool);gradient=copy(grad)
    point=@test_logs WarmupHMC._validated_transport!(rp,old_ir,pool,grad,position,gradient,active)
    @test point.q ≈ [0.4,0.5*exp(0.2)]
    @test position != pool
    @test WarmupHMC.reparam_sources(rp)==[2=>WarmupHMC.PartiallyCentered(0.5)]
end

@testset "a rejected update leaves the run as if the rule kept its controls" begin
    # A selection rule that always moves coordinate 2's centering, against one that
    # never does. With the current point's evaluation poisoned at every move, each
    # move is rejected, so the run must be the never-moving run, draw for draw.
    function sample_with(; move, poisoned)
        poison=Ref(:none)
        calls=Ref(0)
        ir=WarmupHMC.IndexedReparametrization([2=>WarmupHMC.Reparametrization(
            WarmupHMC.PartiallyCentered(0.0),WarmupHMC.PartiallyCentered(0.0),0.0,x->x[1])])
        select!(ir,positions,gradients)=begin
            calls[]+=1
            move || return false
            idx,value=ir.pairs[1]
            ir.pairs[1]=idx=>WarmupHMC.Reparametrization(value.target,
                WarmupHMC.PartiallyCentered(value.source.c==0.0 ? 0.5 : 0.0),value.args...)
            poison[]=poisoned
            true
        end
        rp=WarmupHMC.ReparametrizedProblem(ir,Poisoned(DiagGaussian(zeros(3),[1.0,1.0,0.5]),poison),
            AutoEnzyme();scoring_plan=WarmupHMC.WindowSelectionPlan(select!))
        logs,result=Test.collect_test_logs() do
            adaptive_warmup_mcmc(Xoshiro(2718),rp;n_draws=200,n_evaluations=100,variance_cond_target=1.0,
                init=(;position=[0.1,-0.2,0.3],squared_scale=ones(3)))
        end
        rejections=count(l->occursin("Rejected centering update",l.message),logs)
        (;result,calls=calls[],rejections,sources=WarmupHMC.reparam_sources(rp))
    end
    kept=sample_with(move=false,poisoned=:none)
    rejected=sample_with(move=true,poisoned=:gradient)
    moved=sample_with(move=true,poisoned=:none)
    @test kept.calls>=2
    @test rejected.calls==kept.calls
    @test kept.rejections==0 && moved.rejections==0
    @test rejected.rejections==rejected.calls
    @test rejected.sources==kept.sources==[2=>WarmupHMC.PartiallyCentered(0.0)]
    for field in (:posterior_position,:posterior_gradient,:halo_position,:stepsize,
                  :active_transformation,:n_divergent_samples,:sampling_evaluation_counter,
                  :total_evaluation_counter,:scale_changes)
        @test getfield(rejected.result,field)==getfield(kept.result,field)
    end
    a=kept.result.active_transformation
    E=Matrix{Float64}(I,3,3)
    @test rejected.result.scale_options[a]*E==kept.result.scale_options[a]*E
    # Positive control: accepted moves change the run, so equality above is evidence.
    @test moved.result.posterior_position!=kept.result.posterior_position
end

# Snag find-reparametri-e2a5b2a7: the transport used to copy ~9 d-vectors per
# halo column (2.1 GiB per window end at d = 31,360 with 1000 columns).
@testset "halo transport allocates no per-column vectors" begin
    d=2000
    inner=DiagGaussian(zeros(d),ones(d))
    mk(c)=WarmupHMC.IndexedReparametrization([
        k=>WarmupHMC.Reparametrization(WarmupHMC.PartiallyCentered(1.0),
            WarmupHMC.PartiallyCentered(c),x->x[1],x->x[2]) for k in 3:d])
    rp=WarmupHMC.ReparametrizedProblem(mk(0.7),inner,AutoEnzyme())
    old_ir=mk(0.0)
    rng=Xoshiro(5)
    P=0.3 .* randn(rng,d,64);G=randn(rng,d,64)
    transport!(position,gradient)=WarmupHMC._jointly_transport_halo!(rp,old_ir,P,G,position,gradient)
    transport!(P[:,1:2],G[:,1:2])
    short=(P[:,1:16],G[:,1:16]);long=(copy(P),copy(G))
    bytes_short=@allocated transport!(short...)
    bytes_long=@allocated transport!(long...)
    @test (bytes_long-bytes_short)/48 < 8d/4
    @test long[1][:,1:16]==short[1] && long[2][:,1:16]==short[2]
    # in place, as each column is read before it is written
    inplace=(copy(P),copy(G))
    WarmupHMC._jointly_transport_halo!(rp,old_ir,inplace...,inplace...)
    @test inplace==long
end
end
