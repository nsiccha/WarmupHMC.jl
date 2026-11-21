@testitem "Cooperative second wave recovers on the receiving target" begin
    using LinearAlgebra, LogDensityProblems, Random, WarmupHMC

    # Every borrowed state lies outside the receiving conditional's support.
    # Target-specific additive constants also make raw densities incomparable.
    # Only that slot's own cold start can produce a valid retained transition.
    struct BoundedConditional
        center::Vector{Float64}
    end
    LogDensityProblems.dimension(p::BoundedConditional) = length(p.center)
    LogDensityProblems.capabilities(::Type{BoundedConditional}) = LogDensityProblems.LogDensityOrder{1}()
    function LogDensityProblems.logdensity_and_gradient(p::BoundedConditional, x)
        dx = x - p.center
        maximum(abs, dx) <= 6 || return (-Inf, fill(NaN, length(dx)))
        (1000 * first(p.center) - dot(dx, dx) / 2, -dx)
    end
    LogDensityProblems.logdensity(p::BoundedConditional, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

    targets = [BoundedConditional([20.0i, 0.0]) for i in 1:4]
    starts = [(; position=copy(t.center), squared_scale=ones(2)) for t in targets]
    out = cooperative_warmup_mcmc(Xoshiro(26), targets; init=starts,
        features=[[Float64(i)] for i in eachindex(targets)], n_wave1=1,
        n_draws=1, n_cores=1, n_evaluations=40)

    @test out.stop_reason === :n_draws
    @test out.joint_ess === nothing
    @test getproperty.(out.results, :chain_index) == collect(eachindex(targets))
    @test count(r -> r.accounting.wave == 2, out.results) == 3
    for (target, result) in zip(targets, out.results)
        @test result.status === :done && result.n_samples == 1
        @test length(result.abandoned) == (result.accounting.wave == 2 ? 1 : 0)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(target, only(eachcol(result.posterior_position)))
        @test isfinite(lp)
        @test only(eachcol(result.posterior_gradient)) ≈ gradient
    end
    @test all(r -> r.accounting.independent_probes >= 1, out.results)
end
