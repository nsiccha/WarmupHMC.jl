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
