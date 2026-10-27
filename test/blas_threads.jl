@testitem "samplers run with blas_threads BLAS threads and restore the caller's count" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, LogDensityProblems, Logging

# Focused invocation from the package root:
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "blas_threads.jl"))'

# Multithreaded OpenBLAS starves the gradient of a large posterior under a CPU
# quota or with chains on threads (snag `openblas-threads-2f1bf9e9`), so every
# sampler entry point holds BLAS at `blas_threads` threads (default 1) and gives
# the caller's count back. The count is read from INSIDE the target's gradient:
# completion and cooperative deepcopy the density, so the observations go to a
# module-level set rather than a field.
const SEEN = Set{Int}()
const SEEN_LOCK = ReentrantLock()
struct BLASProbe{P}
    p::P
end
LogDensityProblems.capabilities(::Type{<:BLASProbe}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(b::BLASProbe) = LogDensityProblems.dimension(b.p)
LogDensityProblems.logdensity(b::BLASProbe, x) = LogDensityProblems.logdensity(b.p, x)
LogDensityProblems.logdensity_and_gradient(b::BLASProbe, x) = begin
    lock(() -> push!(SEEN, BLAS.get_num_threads()), SEEN_LOCK)
    LogDensityProblems.logdensity_and_gradient(b.p, x)
end
# Counts seen inside `f()`, and the count after it returned.
observe(f) = begin
    empty!(SEEN)
    f()
    (copy(SEEN), BLAS.get_num_threads())
end

probe = BLASProbe(DiagGaussian([0.0, 1.0, -1.0], [1.0, 2.0, 0.5]))
d = LogDensityProblems.dimension(probe)
init = (; position = zeros(d), squared_scale = ones(d))
caller = 3
saved = BLAS.get_num_threads()
try
    BLAS.set_num_threads(caller)

    @testset "adaptive_warmup_mcmc" begin
        @test observe(() -> adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20)) == (Set([1]), caller)
        @test observe(() -> adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20, blas_threads = 2)) ==
            (Set([2]), caller)
        @test observe(() -> adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20, blas_threads = nothing)) ==
            (Set([caller]), caller)
        # A throw restores the caller's count too.
        @test_throws ErrorException adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20,
            callback = (state, stage) -> error("stop here"))
        @test BLAS.get_num_threads() == caller
    end

    @testset "multi-chain adaptive and resume_warmup_mcmc" begin
        @test observe(() -> adaptive_warmup_mcmc(Xoshiro.(1:2), probe; init, n_draws = 20)) == (Set([1]), caller)
        mktempdir() do dir
            adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20, checkpoint_dir = dir)
            @test observe(() -> adaptive_warmup_mcmc(Xoshiro(1), probe; checkpoint_dir = dir,
                resume = true, n_draws = 40)) == (Set([1]), caller)
            @test observe(() -> Logging.with_logger(Logging.NullLogger()) do
                resume_warmup_mcmc(probe, joinpath(dir, "cp_latest.jls"); n_draws = 60)
            end) == (Set([1]), caller)
        end
    end

    @testset "completion_warmup_mcmc pins once over its chains" begin
        @test observe(() -> Logging.with_logger(Logging.NullLogger()) do
            completion_warmup_mcmc(Xoshiro.(1:2), probe; init, n_draws = 20)
        end) == (Set([1]), caller)
    end

    @testset "cooperative_warmup_mcmc" begin
        @test observe(() -> cooperative_warmup_mcmc(Xoshiro(1), probe; n_chains = 2, init, n_draws = 20)) ==
            (Set([1]), caller)
    end

    @testset "stream_mcmc and fixed_kernel_nuts_step" begin
        mktempdir() do dir
            result = Ref{Any}()
            @test observe(() -> result[] = stream_mcmc(Xoshiro(1), probe, zeros(d); path = joinpath(dir, "s"),
                n_draws = 5, metric = Diagonal(ones(d)), stepsize = 0.5)) == (Set([1]), caller)
            close_stream(result[])
            results = Ref{Any}()
            @test observe(() -> results[] = stream_mcmc(Xoshiro.(1:2), probe, [zeros(d), ones(d)];
                path = joinpath(dir, "m"), n_draws = 5, metric = Diagonal(ones(d)), stepsize = 0.5)) ==
                (Set([1]), caller)
            foreach(close_stream, results[])
        end
        state = FixedKernelNUTSState(Xoshiro(1), zeros(d); metric = Diagonal(ones(d)), stepsize = 0.5)
        @test observe(() -> fixed_kernel_nuts_step(state, probe)) == (Set([1]), caller)
        @test observe(() -> fixed_kernel_nuts_step(state, probe; blas_threads = nothing)) == (Set([caller]), caller)
    end

    @testset "concurrent calls share one pin and restore once" begin
        @test observe(() -> fetch.([Threads.@spawn(adaptive_warmup_mcmc(Xoshiro(i), probe; init, n_draws = 20))
            for i in 1:2])) == (Set([1]), caller)
        # A nested request for a different count runs under the active pin.
        inner = Logging.with_logger(Logging.NullLogger()) do
            WarmupHMC._with_blas_threads(1) do
                WarmupHMC._with_blas_threads(() -> BLAS.get_num_threads(), 2)
            end
        end
        @test inner == 1
        @test BLAS.get_num_threads() == caller
    end

    @testset "invalid blas_threads is an ArgumentError before any work" begin
        for bad in (0, -1, true, 1.5, "1")
            empty!(SEEN)
            @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), probe; init, n_draws = 20,
                blas_threads = bad)
            @test isempty(SEEN)
            @test BLAS.get_num_threads() == caller
        end
    end
finally
    BLAS.set_num_threads(saved)
end
end
