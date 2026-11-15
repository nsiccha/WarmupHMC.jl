@testitem "posterior_log_density is the model's log density at the returned draws" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Serialization
    using LogDensityProblems
    using DifferentiationInterface, Enzyme

# The sampler records the log density at each retained draw as it samples, and
# returns it beside the draws, in the model's own parametrization (snag
# default-starts-t-013b07f7: the completion quorum's mass check reads it).

# A 2-D funnel in its centred (model) parametrization, sampled non-centred:
# y2 = x2 exp(x1).
struct _LogDensityFunnel end
LogDensityProblems.dimension(::_LogDensityFunnel) = 2
LogDensityProblems.capabilities(::Type{_LogDensityFunnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_LogDensityFunnel, y) = -y[1]^2 / 2 - (y[2] * exp(-y[1]))^2 / 2 - y[1]
LogDensityProblems.logdensity_and_gradient(p::_LogDensityFunnel, y) = (
    LogDensityProblems.logdensity(p, y),
    [-y[1] + y[2]^2 * exp(-2y[1]) - 1, -y[2] * exp(-2y[1])],
)
funnel_rp() = ReparametrizedProblem(
    IndexedReparametrization([
        2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 0.0, x -> x[1]),
    ]),
    _LogDensityFunnel(), AutoEnzyme(),
)
model_log_density(p, fit) = [LogDensityProblems.logdensity(p, y) for y in eachcol(fit.posterior_position)]

_blas_threads = LinearAlgebra.BLAS.get_num_threads()
LinearAlgebra.BLAS.set_num_threads(1)
try
    @testset "plain target" begin
        target = DiagGaussian([0.0, 1.0, -2.0], [1.0, 0.3, 2.0])
        fit = adaptive_warmup_mcmc(Xoshiro(5), target; n_draws=200)
        @test length(fit.posterior_log_density) == size(fit.posterior_position, 2)
        @test fit.posterior_log_density ≈ model_log_density(target, fit)
    end

    @testset "reparametrized target: the model frame, without the log-Jacobian" begin
        rp = funnel_rp()
        fit = adaptive_warmup_mcmc(Xoshiro(6), rp; n_draws=200)
        @test length(fit.posterior_log_density) == size(fit.posterior_position, 2)
        @test fit.posterior_log_density ≈ model_log_density(_LogDensityFunnel(), fit)
        # Positive control: the sampling frame's density differs by the
        # log-Jacobian, here the source coordinate 1, which is y1.
        @test !(fit.posterior_log_density ≈ model_log_density(_LogDensityFunnel(), fit) .+
            fit.posterior_position[1, :])
    end

    @testset "checkpoints carry it, and an older checkpoint resumes with it evaluated" begin
        target = DiagGaussian([0.5, -1.0], [1.0, 1.5])
        mktempdir() do dir
            current, older = joinpath(dir, "current"), joinpath(dir, "older")
            adaptive_warmup_mcmc(Xoshiro(8), target; n_draws=100, checkpoint_dir=current)
            cp = deserialize(joinpath(current, "cp_latest.jls"))
            @test size(cp.posterior_position, 2) > 0
            @test length(cp.posterior_log_density) == size(cp.posterior_position, 2)
            @test cp.posterior_log_density ≈
                [LogDensityProblems.logdensity(target, x) for x in eachcol(cp.posterior_position)]
            # The same checkpoint as one written before the key existed.
            mkpath(older)
            serialize(joinpath(older, "cp_latest.jls"),
                Base.structdiff(cp, (; posterior_log_density=nothing)))
            resumed(dir) = resume_warmup_mcmc(target, joinpath(dir, "cp_latest.jls");
                n_draws=200, checkpoint_dir=dir)
            from_current, from_older = resumed(current), resumed(older)
            @test from_older.posterior_position == from_current.posterior_position
            @test from_older.posterior_log_density ≈ from_current.posterior_log_density
            @test from_older.posterior_log_density ≈ model_log_density(target, from_older)
        end
    end
finally
    LinearAlgebra.BLAS.set_num_threads(_blas_threads)
end
end
