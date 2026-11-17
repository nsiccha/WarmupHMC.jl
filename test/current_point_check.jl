@testitem "A target that stops answering finite at the chain's position raises at the next boundary" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LogDensityProblems

# Focused invocation from the package root:
# julia --startup-file=no --project=test -e 'using TestItemRunner; using LinearAlgebra;
# BLAS.set_num_threads(1); TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "current_point_check.jl"))'

# Snag centred-latent-l-16f7838d: after one NaN evaluation at a leapfrog point, an
# RKPPL sampler target returned NaN in part of every later gradient, at every
# position. Its chain stayed at a point the target had answered finite, every
# transition diverged at its first evaluation, the step size fell to 1e-92, and
# the divergence gate blamed the chain's position. Every sampler now evaluates
# the chain's current point once more at each window boundary (adaptive) or round
# barrier (cooperative) and raises when the target no longer answers finite there.

# A standard normal whose gradient turns NaN in coordinate 2 once it has been
# evaluated more than `poison_after` times (0: never). Each copy counts its own
# evaluations, so every chain's copy breaks after its own `poison_after`.
mutable struct Poisoned
    poison_after::Int
    n::Int
end
Poisoned(poison_after) = Poisoned(poison_after, 0)
LogDensityProblems.dimension(::Poisoned) = 3
LogDensityProblems.capabilities(::Type{Poisoned}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::Poisoned, x)
    p.n += 1
    g = -collect(float.(x))
    p.poison_after > 0 && p.n > p.poison_after && (g[2] = NaN)
    -sum(abs2, x) / 2, g
end
LogDensityProblems.logdensity(p::Poisoned, x) = -sum(abs2, x) / 2

init = (; position=zeros(3), squared_scale=ones(3))
captured(f) = try
    f()
    nothing
catch e
    e
end
names_the_change(msg) = occursin("no longer answers finite at the chain's current point", msg) &&
    occursin("1 nonfinite gradient entries (coordinates 2)", msg) &&
    occursin("keeps state between evaluations", msg)

@testset "adaptive: the window that ends after the change raises" begin
    err = captured(() -> adaptive_warmup_mcmc(Xoshiro(1), Poisoned(300); init, n_draws=200, progress=nothing))
    @test err isa ErrorException
    @test names_the_change(err.msg)
    @test occursin("adaptive_warmup_mcmc, at the end of warm-up window 1", err.msg)

    @testset "positive control: the same target unbroken returns its draws" begin
        result = adaptive_warmup_mcmc(Xoshiro(1), Poisoned(0); init, n_draws=200, progress=nothing)
        @test size(result.posterior_position, 2) >= 200
        @test all(isfinite, result.posterior_position)
    end
end

# Two copies of one correlated Gaussian, one of them exempted from the check: the
# check uses no random numbers and records nothing, so both runs agree bit for bit.
struct Checked
    Λ::Matrix{Float64}
end
struct Unchecked
    Λ::Matrix{Float64}
end
const Either = Union{Checked,Unchecked}
LogDensityProblems.dimension(::Either) = 3
LogDensityProblems.capabilities(::Type{<:Either}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity_and_gradient(p::Either, x) = (-x' * p.Λ * x / 2, -(p.Λ * x))
LogDensityProblems.logdensity(p::Either, x) = -x' * p.Λ * x / 2
WarmupHMC._check_current_point(::Unchecked, point, where) = nothing
correlated = let A = [1.0 0.0 0.0; 0.9 0.5 0.0; 0.0 0.3 2.0]
    inv(A * A')
end

@testset "the check changes no draw, step size or counter of a run" begin
    # `variance_cond_target = 1` restarts every window, so the check also runs
    # before every restart's re-adaptation.
    for kw in ((; n_draws=200), (; n_draws=200, variance_cond_target=1.0))
        a = adaptive_warmup_mcmc(Xoshiro(2), Checked(correlated); init, progress=nothing, kw...)
        b = adaptive_warmup_mcmc(Xoshiro(2), Unchecked(correlated); init, progress=nothing, kw...)
        @test a.posterior_position == b.posterior_position
        @test a.stepsize == b.stepsize
        @test a.total_evaluation_counter == b.total_evaluation_counter
        @test a.sampling_evaluation_counter == b.sampling_evaluation_counter
    end
    a = cooperative_warmup_mcmc(Xoshiro(6), Checked(correlated); n_chains=2, init, n_draws=200, progress=nothing)
    b = cooperative_warmup_mcmc(Xoshiro(6), Unchecked(correlated); n_chains=2, init, n_draws=200, progress=nothing)
    @test [r.posterior_position for r in a.results] == [r.posterior_position for r in b.results]
    @test a.total_evaluation_counter == b.total_evaluation_counter
end

@testset "completion: the broken chain fails with the message, the others complete" begin
    err = captured(() -> completion_warmup_mcmc([Xoshiro(3), Xoshiro(4)], [Poisoned(300), Poisoned(0)];
        min_completed=2, init, n_draws=200, progress=nothing))
    @test err isa WarmupHMC.CompletionQuorumError
    # The quorum error carries the failed chain's first line.
    @test occursin("; chain 1 failed: adaptive_warmup_mcmc, at the end of warm-up window",
                   sprint(showerror, err))
    chains = err.outcome.completion.chains
    @test chains[1].status === :failed
    @test names_the_change(sprint(showerror, chains[1].error.ex))
    @test chains[2].status === :completed
end

@testset "cooperative: the round in which a chain's target breaks raises" begin
    err = captured(() -> cooperative_warmup_mcmc(Xoshiro(5), Poisoned(300); n_chains=2, init,
        n_draws=200, progress=nothing))
    @test err isa ErrorException
    @test names_the_change(err.msg)
    @test occursin("cooperative_warmup_mcmc, chain 1 at the end of round 1", err.msg)

    @testset "positive control: the same target unbroken stops on n_draws" begin
        out = cooperative_warmup_mcmc(Xoshiro(5), Poisoned(0); n_chains=2, init, n_draws=200, progress=nothing)
        @test out.stop_reason === :n_draws
    end
end
end
