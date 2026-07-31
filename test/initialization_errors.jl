# The initialization error contract.
#
# A non-finite log density at the starting point used to surface as
# `AssertionError: length(init.elbo_estimates) > 0` — five frames downstream of the
# fault, naming an ELBO vector rather than the density. This file pins the two
# checks that replaced it, and the reason both have to exist: they fire on
# DIFFERENT causes, so neither makes the other redundant.
#
# Every negative check below is paired with a positive one. A check that fired
# unconditionally, or a message builder that threw while building its own message,
# would satisfy "an error was raised" just as well — so each testset also asserts
# that the healthy shape of the same input still initializes.

"""
    _with_empty_elbo(result)

Rebuild `result` with an empty `elbo_estimates` and every other field unchanged.
Field-NAME driven rather than positional, so it survives a `PathfinderResult`
layout change instead of silently mis-assigning fields.
"""
_with_empty_elbo(result) = WarmupHMC.Pathfinder.PathfinderResult(
    (field === :elbo_estimates ? empty(getfield(result, field)) : getfield(result, field)
     for field in fieldnames(typeof(result)))...
)

_init_error(f) = try
    f()
    nothing
catch err
    err
end

@testset "non-finite log density at init" begin
    problem = NaNProblem(4)

    err = _init_error() do
        WarmupHMC.initialize_mcmc(problem, zeros(4); rng=Xoshiro(20260731), progress=nothing)
    end
    @test err isa ErrorException
    message = sprint(showerror, err)

    # It is named as an INITIALIZATION failure, with the offending value...
    @test occursin("Initialization failed", message)
    @test occursin("the log density and its gradient are not finite", message)
    @test occursin("log density at init: NaN", message)
    @test occursin("4 of 4 components non-finite", message)
    # ...and explains why the optimizer cannot proceed from it.
    @test occursin("iteration 0", message)
    # It is NOT the ELBO diagnostic this replaced.
    @test !(err isa AssertionError)
    @test !occursin("elbo_estimates", message)
end

@testset "non-finite gradient at init, finite log density" begin
    # NOT centred on the starting point: L-BFGS declares convergence at iteration 0
    # from an already-stationary start, which produces an empty `elbo_estimates` and
    # would make the positive control below fail for an unrelated reason.
    inner = DiagGaussian([1.0, -2.0, 0.5], [1.0, 2.0, 0.5])
    problem = BadGradientProblem(inner, 2)

    # Positive control: the unpoisoned problem initializes from the same point.
    @test WarmupHMC.initialize_mcmc(
        inner, zeros(3); rng=Xoshiro(20260731), progress=nothing) isa NamedTuple

    err = _init_error() do
        WarmupHMC.initialize_mcmc(problem, zeros(3); rng=Xoshiro(20260731), progress=nothing)
    end
    @test err isa ErrorException
    message = sprint(showerror, err)
    # The headline names the GRADIENT, not the log density, which is finite here.
    @test occursin("the gradient of the log density is not finite", message)
    @test occursin("1 of 3 components non-finite", message)
    @test occursin("[2] = Inf", message)
end

@testset "empty elbo_estimates" begin
    problem = DiagGaussian([0.0, 1.0], [1.0, 2.0])
    result = WarmupHMC.mypathfinder(
        problem; rng=Xoshiro(20260731), init=zeros(2), maxiters=50)

    # Positive control: a real Pathfinder result on this problem is non-empty and
    # initializes. Without it, a check that always fired would pass below.
    @test !isempty(result.elbo_estimates)
    @test WarmupHMC.initialize_mcmc(problem, result) isa NamedTuple

    err = _init_error() do
        WarmupHMC.initialize_mcmc(problem, _with_empty_elbo(result))
    end
    @test err isa ErrorException
    message = sprint(showerror, err)
    @test occursin("Initialization failed", message)
    @test occursin("no ELBO estimates", message)
    # The residual causes this message exists for — it is NOT made redundant by
    # the non-finite check, which cannot explain a finite-start failure.
    @test occursin("line search", message)
    @test !(err isa AssertionError)
end

@testset "the retry loop rethrows the explicit error, not the assertion" begin
    # `initialize_mcmc(lpdf, ::Distribution)` swallows and retries `ntries` times,
    # then rethrows. What the user finally sees must be the actionable message.
    err = _init_error() do
        WarmupHMC.initialize_mcmc(
            NaNProblem(2), Uniform(-2.0, 2.0);
            rng=Xoshiro(20260731), progress=nothing, ntries=2)
    end
    @test err isa ErrorException
    @test occursin("Initialization failed", sprint(showerror, err))
end
