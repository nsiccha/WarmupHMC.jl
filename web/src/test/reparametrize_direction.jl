@testitem "reparametrize! reports draws in model coordinates" setup=[Determinism] tags=[:reparametrization] begin
    using WarmupHMC, LogDensityProblems
    using WarmupHMC: IndexedReparametrization, Reparametrization, PartiallyCentered,
                     reparametrize!, ReparametrizedProblem

    # Minimal inner problem — only its coordinate convention matters here.
    struct _DirProbe end
    LogDensityProblems.logdensity(::_DirProbe, v) = 0.0
    LogDensityProblems.dimension(::_DirProbe) = 3
    LogDensityProblems.capabilities(::Type{_DirProbe}) = LogDensityProblems.LogDensityOrder{0}()

    # Regression guard for the back-transform DIRECTION.
    #
    # `golden_awm.jl` compares against a frozen baseline, so it pins whatever values
    # the code happened to produce — it cannot notice an inverted transform. This
    # test instead asserts the invariant: the draws handed back to the caller must be
    # the same coordinates the inner problem was actually evaluated at.
    #
    # The bug this guards: `reparametrize!` applied `inverse(ir)` (target -> source)
    # to draws already stored in `source`, yielding -0.5 where the model saw 3.5.
    # It was invisible whenever adaptation left `source == target` (both identity).
    @testset "reparametrize! reports draws in model coordinates" begin
        ir = IndexedReparametrization([
            1 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0),
                                   x -> x[2], x -> x[3])
        ])
        x = [0.5, 2.0, log(3.0)]        # a sampler draw, in `source` coordinates
        _, y = ir(x)                    # exactly what `logdensity` feeds the inner problem

        M = reshape(copy(x), :, 1)
        reparametrize!(ReparametrizedProblem(ir, _DirProbe()), M)

        @test M[:, 1] ≈ y
        @test M[1, 1] ≈ 3.5             # not -0.5, which the inverted direction gave

        # source == target must stay an exact no-op.
        ir0 = IndexedReparametrization([
            1 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                                   x -> x[2], x -> x[3])
        ])
        M0 = reshape(copy(x), :, 1)
        reparametrize!(ReparametrizedProblem(ir0, _DirProbe()), M0)
        @test M0[:, 1] == x
    end
end
