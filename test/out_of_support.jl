@testitem "Targets outside their support" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "out_of_support.jl"))'

# The out-of-support contract a consumer's target relies on (warmuphmc-use §1a,
# snag `warmuphmc-use-ta-7fc2f0ef`):
#
# * a leapfrog position whose log density is non-finite (-Inf, NaN, +Inf, or a
#   finite value with a non-finite gradient) is a divergent leaf, and its
#   gradient is never read — so every such answer gives the same run;
# * an exception thrown by the target is never caught while sampling;
# * a non-finite START is an error, never a divergence.

# A standard normal truncated to x[1] > 0. Its mode sits ON the support boundary,
# so leapfrog steps leave the support constantly. `answer` is what it returns
# outside the support; `calls` counts those evaluations, so a run that never left
# the support cannot pass the identity checks vacuously.
struct TruncatedNormal{F}
    answer::F
    calls::Base.RefValue{Int}
end
TruncatedNormal(answer) = TruncatedNormal(answer, Ref(0))
LogDensityProblems.dimension(::TruncatedNormal) = 3
LogDensityProblems.capabilities(::Type{<:TruncatedNormal}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::TruncatedNormal, x)
    x[1] > 0 && return (-sum(abs2, x) / 2, -collect(float.(x)))
    p.calls[] += 1
    p.answer(x)
end
LogDensityProblems.logdensity(p::TruncatedNormal, x) =
    first(LogDensityProblems.logdensity_and_gradient(p, x))

answers = (
    neginf_zero = x -> (-Inf, zero(float.(x))),
    neginf_nan = x -> (-Inf, fill(NaN, length(x))),
    neginf_prior = x -> (-Inf, -collect(float.(x))),  # a prior-only gradient
    nan = x -> (NaN, fill(NaN, length(x))),
    posinf = x -> (Inf, zero(float.(x))),
    nan_gradient = x -> (-sum(abs2, x) / 2, fill(NaN, length(x))),
)
throwing(x) = throw(DomainError(x[1], "x[1] must be positive"))
inside = [0.5, 0.0, 0.0]
outside = [-0.5, 0.0, 0.0]
init = (; position=inside, squared_scale=ones(3))

# One run per answer; each returns (fingerprint, divergence count, out-of-support calls).
function runs(f)
    map(answers) do answer
        p = TruncatedNormal(answer)
        out = f(p)
        (; out..., calls=p.calls[])
    end
end
function check_same_run(results)
    reference = first(results)
    @test reference.calls > 0
    @test reference.n_divergent > 0
    for (name, r) in pairs(results)
        @test r.calls == reference.calls
        @test r.n_divergent == reference.n_divergent
        @test r.fingerprint == reference.fingerprint
    end
end

blas_threads = BLAS.get_num_threads()
BLAS.set_num_threads(1)
try
    @testset "a leaf outside the support is a divergence, whatever the gradient" begin
        @testset "adaptive_warmup_mcmc" begin
            results = runs() do p
                fit = adaptive_warmup_mcmc(Xoshiro(1), p; init=init, n_draws=200)
                @test all(>(0), fit.posterior_position[1, :])
                (; fingerprint=(Matrix(fit.posterior_position), fit.stepsize),
                   n_divergent=fit.n_divergent_samples)
            end
            check_same_run(results)
        end
        @testset "fixed_kernel_nuts_step" begin
            results = runs() do p
                state = FixedKernelNUTSState(Xoshiro(7), inside; metric=ones(3), stepsize=1.5)
                positions = Vector{Float64}[]
                n_divergent = 0
                for _ in 1:100
                    step = fixed_kernel_nuts_step(state, p)
                    state = step.state
                    n_divergent += step.divergent
                    push!(positions, copy(state.position))
                end
                @test all(q -> q[1] > 0, positions)
                (; fingerprint=positions, n_divergent)
            end
            check_same_run(results)
        end
        @testset "stream_mcmc" begin
            results = runs() do p
                result = stream_mcmc(Xoshiro(7), p, inside;
                    path=joinpath(mktempdir(), "stream"), n_draws=100,
                    metric=ones(3), stepsize=1.5)
                try
                    @test all(>(0), result.draws[1, :])
                    @test count(!iszero, result.divergences) == result.n_divergent
                    (; fingerprint=copy(result.draws), n_divergent=result.n_divergent)
                finally
                    close_stream(result)
                end
            end
            check_same_run(results)
        end
    end

    @testset "a thrown error is never caught while sampling" begin
        p = TruncatedNormal(throwing)
        @test_throws DomainError adaptive_warmup_mcmc(Xoshiro(1), p; init=init, n_draws=200)
        @test p.calls[] == 1
        steps(p) = let state = FixedKernelNUTSState(Xoshiro(7), inside; metric=ones(3), stepsize=1.5)
            for _ in 1:100
                state = fixed_kernel_nuts_step(state, p).state
            end
        end
        @test_throws DomainError steps(p)
        @test_throws DomainError stream_mcmc(Xoshiro(7), p, inside;
            path=joinpath(mktempdir(), "stream"), n_draws=100, metric=ones(3), stepsize=1.5)
        # Positive control: the same calls complete once the target answers -Inf.
        ok = TruncatedNormal(answers.neginf_zero)
        @test size(adaptive_warmup_mcmc(Xoshiro(1), ok; init=init, n_draws=200).posterior_position, 2) >= 200
    end

    @testset "a start outside the support is an error, never a divergence" begin
        for answer in (answers.neginf_zero, answers.neginf_nan)
            @test_throws WarmupHMC.DynamicHMC.DynamicHMCError adaptive_warmup_mcmc(
                Xoshiro(1), TruncatedNormal(answer);
                init=(; position=outside, squared_scale=ones(3)), n_draws=10)
            @test_throws ArgumentError fixed_kernel_nuts_step(
                FixedKernelNUTSState(Xoshiro(7), outside; metric=ones(3), stepsize=0.5),
                TruncatedNormal(answer))
            @test_throws ArgumentError stream_mcmc(Xoshiro(7), TruncatedNormal(answer), outside;
                path=joinpath(mktempdir(), "stream"), n_draws=10, metric=ones(3), stepsize=0.5)
        end
        # NaN, +Inf and a non-finite gradient fail DynamicHMC's strict start evaluation.
        for answer in (answers.nan, answers.posinf, answers.nan_gradient)
            @test_throws WarmupHMC.DynamicHMC.DynamicHMCError fixed_kernel_nuts_step(
                FixedKernelNUTSState(Xoshiro(7), outside; metric=ones(3), stepsize=0.5),
                TruncatedNormal(answer))
        end
        # Positive control: the same start one step inside the support samples.
        @test fixed_kernel_nuts_step(
            FixedKernelNUTSState(Xoshiro(7), inside; metric=ones(3), stepsize=0.5),
            TruncatedNormal(answers.neginf_zero)).state.position[1] > 0
    end
finally
    BLAS.set_num_threads(blas_threads)
end
end
