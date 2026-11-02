@testitem "ReparametrizableDistributions integration parity" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LogDensityProblems
    using DifferentiationInterface, Enzyme
    import ReparametrizableDistributions as RD

    # Freeze the former WarmupHMC implementation's outputs, not a second copy
    # of its formula. Captured at d10b54417ad8fa6e39bacfa7b377d436efdcfe2e with
    # Julia 1.12.7 and the seeded Enzyme dogfood build before importing RD's primitive.
    struct QuarticTarget end
    LogDensityProblems.dimension(::QuarticTarget) = 5
    LogDensityProblems.capabilities(::Type{QuarticTarget}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::QuarticTarget, x) = -sum(z -> z^2 / 2 + z^4 / 4, x)
    LogDensityProblems.logdensity_and_gradient(p::QuarticTarget, x) =
        (LogDensityProblems.logdensity(p, x), -x .- x.^3)

    @test PartiallyCentered === RD.PartiallyCentered
    @test WarmupHMC.reparam === RD.reparam
    @test Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt) !== nothing

    cases = [
        (ct = 1.0, cs = 0.0, x = [0.7, 0.0, 0.0, 0.0, 0.0], ljac = 0.0, y = [0.7, 0.7, 0.7, 0.7, 0.0], ld = -1.2200999999999997, g = [-4.172, -1.043, -1.043, -1.043, 1.5]),
        (ct = 1.0, cs = 0.5, x = [0.8, 0.4, 0.4, 0.4, 0.25], ljac = 0.1875, y = [0.8, 0.8, 0.8, 0.8, 0.25], ld = -1.5343265625000004, g = [-3.1530749048496536, -1.3966167301002317, -1.3966167301002317, -1.3966167301002317, 0.484375]),
        (ct = 1.0, cs = 1.0, x = [0.7, 0.7, 0.7, 0.7, 0.2], ljac = 0.0, y = [0.7, 0.7, 0.7, 0.7, 0.2], ld = -1.2404999999999997, g = [-1.043, -1.043, -1.043, -1.043, -0.20800000000000002]),
        (ct = 1.0, cs = 0.35, x = [-0.4, 0.3, -0.8, 1.1, -0.7], ljac = -0.6824999999999999, y = [-0.4, -0.04953013454110361, -0.9257047981883447, 0.58768780265689, -0.7], ld = -1.8897095828213912, g = [1.169320407770879, 0.03954864736170135, 1.369196897012813, -0.6297797125156427, 1.476161498019671]),
        (ct = 0.25, cs = 1.0, x = [-0.4, 0.3, -0.8, 1.1, -0.7], ljac = 0.7874999999999999, y = [-0.4, 0.8101235277179437, -0.6200705872673964, 1.8502647022527365, -0.7], ld = -4.910751614530708, g = [9.566862024292126, -1.7445868020767261, 1.1161754914965911, -10.641434620751806, 6.529187333442521]),
    ]
    @testset "source $(row.cs), target $(row.ct)" for row in cases
        ir = IndexedReparametrization([i => Reparametrization(
            PartiallyCentered(row.ct), PartiallyCentered(row.cs),
            q -> q[1], q -> q[5] / 2) for i in 2:4])
        rp = ReparametrizedProblem(ir, QuarticTarget(), AutoEnzyme())
        x = copy(row.x)
        ljac, y = ir(x)
        ld, g = LogDensityProblems.logdensity_and_gradient(rp, x)
        @test ljac ≈ row.ljac atol = 1e-14
        @test y ≈ row.y rtol = 1e-14
        @test ld ≈ row.ld rtol = 1e-14
        @test g ≈ row.g rtol = 1e-13
        @test LogDensityProblems.logdensity(rp, x) ≈ row.ld rtol = 1e-14
        @test g ≈ fd_gradient(q -> LogDensityProblems.logdensity(rp, q), x) rtol = 1e-6 atol = 1e-9
        @test x == row.x
    end
end
