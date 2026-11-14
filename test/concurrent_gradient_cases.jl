# Cases for `concurrent_gradients.jl`, run in a child Julia with two threads: the
# refusal fires only when chains can run at once, and the suite itself usually
# runs on one thread.
module ConcurrentGradientCases

using Test, WarmupHMC, Random, LogDensityProblems, LinearAlgebra
const WH = WarmupHMC

struct Gaussian
    d::Int
end
LogDensityProblems.capabilities(::Type{Gaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::Gaussian) = p.d
LogDensityProblems.logdensity(::Gaussian, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::Gaussian, x) = (-sum(abs2, x) / 2, -x)

# An AD backend whose gradients are not safe beside other threads, as an Enzyme
# without the concurrent-GC repair. Reaching its gradient raises
# `GradientReached`, which shows that a call got past the refusal.
struct UnsafeBackend end
struct GradientReached <: Exception end
WH._unsafe_concurrent_gradients(::UnsafeBackend) = "The test backend is unsafe."
WH._reparam_gradient(::UnsafeBackend, ir, x, y, g_y) = throw(GradientReached())

const D = 3
reparametrized(pairs) = ReparametrizedProblem(IndexedReparametrization(pairs), Gaussian(D), UnsafeBackend())
const PAIR = 1 => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), 0.0, x -> x[3])
unsafe() = reparametrized([PAIR])
empty_map() = reparametrized(typeof(PAIR)[])
const INIT = (; position=zeros(D), squared_scale=ones(D))
const RUN = (; n_draws=20, nonlinear_adapt=false, init=INIT)

refused(f) = try
    f()
    false
catch e
    e isa ArgumentError && occursin("The test backend is unsafe.", e.msg)
end
reached(f) = try
    f()
    false
catch e
    e isa GradientReached || (e isa CompositeException && all(c -> c isa TaskFailedException &&
        c.task.result isa GradientReached, e.exceptions))
end
refusal_message(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : ""
end

function run_cases()
    @testset "concurrent chains with an unsafe reparametrization backend" begin
        @test Threads.nthreads() == 2
        rngs = [Xoshiro(i) for i in 1:2]
        mktempdir() do root
            dir = joinpath(root, "adaptive")
            @test refused(() -> adaptive_warmup_mcmc(rngs, unsafe(); RUN..., checkpoint_dir=dir))
            @test refused(() -> adaptive_warmup_mcmc(rngs, [unsafe(), unsafe()]; RUN...))
            @test !ispath(dir)
            msg = refusal_message(() -> adaptive_warmup_mcmc(rngs, unsafe(); RUN...))
            @test occursin("adaptive_warmup_mcmc would run 2 chains at once", msg)
            @test occursin("https://github.com/nsiccha/Enzyme.jl", msg)
            @test occursin("72d5d25", msg)
            dir = joinpath(root, "completion")
            @test refused(() -> completion_warmup_mcmc(rngs, unsafe(); RUN..., checkpoint_dir=dir))
            @test !ispath(dir)
            dir = joinpath(root, "cooperative")
            @test refused(() -> cooperative_warmup_mcmc(Xoshiro(1), unsafe(); n_chains=2, n_draws=20,
                nonlinear_adapt=false, init=INIT, checkpoint_dir=dir))
            @test !ispath(dir)
            @test refused(() -> stream_mcmc(rngs, unsafe(), [zeros(D), zeros(D)];
                path=joinpath(root, "stream"), n_draws=2, metric=Diagonal(ones(D)), stepsize=0.5))
            @test refused(() -> resume_warmup_mcmc([unsafe(), unsafe()], joinpath(root, "nothing")))
        end
    end
    @testset "chains that do not run at once, or take no guarded gradient, are not refused" begin
        rngs = [Xoshiro(i) for i in 1:2]
        @test reached(() -> adaptive_warmup_mcmc(rngs, unsafe(); RUN..., parallel=false))
        @test reached(() -> cooperative_warmup_mcmc(Xoshiro(1), unsafe(); n_chains=2, n_cores=1,
            n_draws=20, nonlinear_adapt=false, init=INIT))
        @test reached(() -> cooperative_warmup_mcmc(Xoshiro(1), unsafe(); n_chains=1, n_draws=20,
            nonlinear_adapt=false, init=INIT))
        mktempdir() do root
            @test reached(() -> stream_mcmc(rngs, unsafe(), [zeros(D), zeros(D)];
                path=joinpath(root, "stream"), n_draws=2, metric=Diagonal(ones(D)), stepsize=0.5, parallel=false))
        end
        # An empty map takes no gradient of a reparametrization.
        @test reached(() -> adaptive_warmup_mcmc(rngs, empty_map(); RUN...))
        # A plain density's gradient is its own.
        out = adaptive_warmup_mcmc(rngs, Gaussian(D); RUN...)
        @test length(out) == 2
        @test all(r -> size(r.posterior_position, 2) >= 20, out)
    end
end

end
