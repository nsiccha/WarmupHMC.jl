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
# outside the support; `calls(p)` counts those evaluations, so a run that never
# left the support cannot pass the identity checks vacuously. The counter lives in
# `call_counts`, not in the struct: cooperative and clustered `deepcopy` the target
# per chain, and a copied counter would leave the caller's at zero.
call_counts = Threads.Atomic{Int}[]
struct TruncatedNormal{F}
    answer::F
    id::Int
end
TruncatedNormal(answer) = (push!(call_counts, Threads.Atomic{Int}(0));
    TruncatedNormal(answer, length(call_counts)))
calls(p::TruncatedNormal) = call_counts[p.id][]
LogDensityProblems.dimension(::TruncatedNormal) = 3
LogDensityProblems.capabilities(::Type{<:TruncatedNormal}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::TruncatedNormal, x)
    x[1] > 0 && return (-sum(abs2, x) / 2, -collect(float.(x)))
    Threads.atomic_add!(call_counts[p.id], 1)
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
# A target whose support excludes every position (a resumed stream's new target).
struct Excluded end
LogDensityProblems.dimension(::Excluded) = 3
LogDensityProblems.capabilities(::Type{Excluded}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(::Excluded, x) = (-Inf, zero(float.(x)))
LogDensityProblems.logdensity(::Excluded, x) = -Inf
inside = [0.5, 0.0, 0.0]
outside = [-0.5, 0.0, 0.0]
init = (; position=inside, squared_scale=ones(3))

# One run per answer; each returns (fingerprint, divergence count, out-of-support calls).
function runs(f)
    map(answers) do answer
        p = TruncatedNormal(answer)
        out = f(p)
        (; out..., calls=calls(p))
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
        # Two chains each; `results` is in completion order, so compare by identity.
        pooled(out) = sort(out.results; by=c -> c.chain_index)
        @testset "cooperative_warmup_mcmc" begin
            results = runs() do p
                chains = pooled(cooperative_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
                    init=init, n_draws=200, n_evaluations_budget=20000))
                (; fingerprint=[Matrix(c.posterior_position) for c in chains],
                   n_divergent=sum(c.n_divergent_samples for c in chains))
            end
            check_same_run(results)
        end
        @testset "clustered_warmup_mcmc" begin
            results = runs() do p
                chains = clustered_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
                    init=init, n_draws=200).results
                (; fingerprint=[Matrix(c.posterior_position) for c in chains],
                   n_divergent=sum(c.n_divergent_samples for c in chains))
            end
            check_same_run(results)
        end
    end

    @testset "a thrown error is never caught while sampling" begin
        p = TruncatedNormal(throwing)
        @test_throws DomainError adaptive_warmup_mcmc(Xoshiro(1), p; init=init, n_draws=200)
        @test calls(p) == 1
        steps(p) = let state = FixedKernelNUTSState(Xoshiro(7), inside; metric=ones(3), stepsize=1.5)
            for _ in 1:100
                state = fixed_kernel_nuts_step(state, p).state
            end
        end
        @test_throws DomainError steps(p)
        @test_throws DomainError stream_mcmc(Xoshiro(7), p, inside;
            path=joinpath(mktempdir(), "stream"), n_draws=100, metric=ones(3), stepsize=1.5)
        # Chains run as tasks surface it wrapped (`CompositeException`/`TaskFailedException`).
        rootcause(e::CompositeException) = rootcause(first(e.exceptions))
        rootcause(e::TaskFailedException) = rootcause(e.task.exception)
        rootcause(e) = e
        err = try
            cooperative_warmup_mcmc([Xoshiro(11), Xoshiro(12)], p;
                init=init, n_draws=200, n_evaluations_budget=20000)
        catch e
            e
        end
        @test rootcause(err) isa DomainError
        # Positive control: the same calls complete once the target answers -Inf.
        ok = TruncatedNormal(answers.neginf_zero)
        @test size(adaptive_warmup_mcmc(Xoshiro(1), ok; init=init, n_draws=200).posterior_position, 2) >= 200
    end

    @testset "a start outside the support is an error, never a divergence" begin
        start_error(f) = try
            f()
            nothing
        catch err
            err
        end
        for answer in (answers.neginf_zero, answers.neginf_nan)
            @test_throws WarmupHMC.DynamicHMC.DynamicHMCError adaptive_warmup_mcmc(
                Xoshiro(1), TruncatedNormal(answer);
                init=(; position=outside, squared_scale=ones(3)), n_draws=10)
            err = start_error() do
                fixed_kernel_nuts_step(
                    FixedKernelNUTSState(Xoshiro(7), outside; metric=ones(3), stepsize=0.5),
                    TruncatedNormal(answer))
            end
            @test err isa ArgumentError
            @test occursin("at the state's position, so it lies outside the target's support", err.msg)
            err = start_error() do
                stream_mcmc(Xoshiro(7), TruncatedNormal(answer), outside;
                    path=joinpath(mktempdir(), "stream"), n_draws=10, metric=ones(3), stepsize=0.5)
            end
            @test err isa ArgumentError
            @test occursin("at the start position, so it lies outside the target's support", err.msg)
        end
        # A resumed stream re-evaluates its last stored draw under the new target.
        path = joinpath(mktempdir(), "stream")
        close_stream(stream_mcmc(Xoshiro(7), TruncatedNormal(answers.neginf_zero), inside;
            path, n_draws=5, metric=ones(3), stepsize=0.5))
        err = start_error() do
            stream_mcmc(Excluded(); path, n_draws=10, metric=ones(3), stepsize=0.5)
        end
        @test err isa ArgumentError
        @test occursin("at the last stored draw (column 5 of", err.msg)
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
