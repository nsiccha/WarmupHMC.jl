# Unit tests for src/WrappedLogDensityProblems.jl (the forwarding wrappers)
# and the ext/DifferentiationInterfaceExt.jl gradient path.

const NamedPosterior = WarmupHMC.NamedPosterior
const CountingPosterior = WarmupHMC.CountingPosterior
const count_and_time = WarmupHMC.count_and_time

@testset "NamedPosterior forwarding" begin
    n = 3
    inner = DiagGaussian(zeros(n), ones(n))
    np = NamedPosterior(inner, "my-normal")
    x = randn(n)
    @test LogDensityProblems.dimension(np) == n
    @test LogDensityProblems.capabilities(typeof(np)) == LogDensityProblems.capabilities(typeof(inner))
    @test LogDensityProblems.logdensity(np, x) == LogDensityProblems.logdensity(inner, x)
    @test LogDensityProblems.logdensity_and_gradient(np, x) == LogDensityProblems.logdensity_and_gradient(inner, x)
    @test string(np) == "my-normal"
end

@testset "CountingPosterior counts gradient calls" begin
    n = 3
    inner = DiagGaussian(zeros(n), ones(n))
    cp = CountingPosterior(inner)
    x = randn(n)
    @test cp.count[] == 0
    LogDensityProblems.logdensity(cp, x)                  # value-only ⇒ no increment
    @test cp.count[] == 0
    LogDensityProblems.logdensity_and_gradient(cp, x)     # gradient ⇒ increment
    LogDensityProblems.logdensity_and_gradient(cp, x)
    @test cp.count[] == 2
    @test LogDensityProblems.logdensity_and_gradient(cp, x) == LogDensityProblems.logdensity_and_gradient(inner, x)
end

@testset "count_and_time" begin
    n = 3
    inner = DiagGaussian(zeros(n), ones(n))
    x = randn(n)
    out = count_and_time(inner) do wrapped
        for _ in 1:5
            LogDensityProblems.logdensity_and_gradient(wrapped, x)
        end
        :done
    end
    @test out.n_evaluations == 5
    @test out.result == :done
    @test out.elapsed ≥ 0
end

@testset "DifferentiationInterfaceExt gradient path" begin
    n = 5
    inner = DiagGaussian(randn(n), rand(n) .+ 0.5)
    x = randn(n)
    # identity reparam ⇒ the custom gradient equals the inner problem's gradient
    rp0 = WarmupHMC.ReparametrizedProblem(WarmupHMC.IndexedReparametrization([]), inner, AutoForwardDiff())
    ld0, g0 = LogDensityProblems.logdensity_and_gradient(rp0, x)
    ldi, gi = LogDensityProblems.logdensity_and_gradient(inner, x)
    @test ld0 ≈ ldi
    @test g0 ≈ gi
    # non-trivial reparam ⇒ gradient matches finite differences of the reparametrized logdensity
    loc_idx, scale_idx = 4, 5
    ir = WarmupHMC.IndexedReparametrization([
        i => WarmupHMC.Reparametrization(WarmupHMC.PartiallyCentered(0.0), WarmupHMC.PartiallyCentered(1.0),
                                         x -> x[loc_idx], x -> x[scale_idx]) for i in 1:3])
    rp = WarmupHMC.ReparametrizedProblem(ir, inner, AutoForwardDiff())
    ld, g = LogDensityProblems.logdensity_and_gradient(rp, x)
    @test ld ≈ LogDensityProblems.logdensity(rp, x)
    @test g ≈ fd_gradient(z -> LogDensityProblems.logdensity(rp, z), x) rtol = 1e-4
end
