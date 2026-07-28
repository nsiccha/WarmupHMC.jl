@testitem "multi-chain lpdf isolation" setup=[Determinism] tags=[:sampler] begin
    using WarmupHMC, Random, LogDensityProblems

    # Regression guard for the scalar-lpdf multi-chain entry point
    # (`adaptive_warmup_mcmc(rngs::AbstractArray, lpdf; ...)`).
    #
    # It used to build the per-chain vector with `fill(lpdf, size(rngs))`, which
    # stores the SAME object in every slot. A stateful lpdf was then shared by every
    # chain — notably a `ReparametrizedProblem`, whose `IndexedReparametrization` is
    # mutated in place by `optimize!` at each window boundary while the gradient hot
    # path reads it. Under the default `parallel=true` that is a data race on a
    # `Vector{Pair{Int,Reparametrization}}`; single-threaded it is still wrong, since
    # chain i>1 samples under whatever centering chain i-1 last wrote. Both silent.
    #
    # The contract asserted here is the one `cooperative_warmup_mcmc` and
    # `clustered_warmup_mcmc` already keep: every chain gets its own `deepcopy`, and
    # the caller's object is never touched. Stated as object identity rather than as
    # reparametrization values, so the guard needs no AD backend and cannot be
    # satisfied by a target that happens to adapt to the same answer on every chain.

    # Ill-conditioned diagonal Gaussian — supplies its own gradient, so nothing here
    # needs ForwardDiff / DifferentiationInterface.
    struct _IsoGauss{V}
        sigma::V
    end
    LogDensityProblems.dimension(g::_IsoGauss) = length(g.sigma)
    LogDensityProblems.capabilities(::Type{<:_IsoGauss}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(g::_IsoGauss, x) = -sum(abs2, x ./ g.sigma) / 2
    LogDensityProblems.logdensity_and_gradient(g::_IsoGauss, x) =
        (LogDensityProblems.logdensity(g, x), -x ./ g.sigma .^ 2)

    # Records which lpdf OBJECT each gradient evaluation ran against. The sink is a
    # module-level global rather than a field, so `deepcopy` cannot duplicate it —
    # every copy of the probe still reports into the same list.
    const _ISO_SEEN = UInt[]
    const _ISO_LOCK = ReentrantLock()
    struct _IdProbe{P}
        inner::P
    end
    LogDensityProblems.dimension(p::_IdProbe) = LogDensityProblems.dimension(p.inner)
    LogDensityProblems.capabilities(::Type{<:_IdProbe{P}}) where {P} = LogDensityProblems.capabilities(P)
    LogDensityProblems.logdensity(p::_IdProbe, x) = LogDensityProblems.logdensity(p.inner, x)
    LogDensityProblems.logdensity_and_gradient(p::_IdProbe, x) = begin
        lock(_ISO_LOCK) do
            push!(_ISO_SEEN, objectid(p))
        end
        LogDensityProblems.logdensity_and_gradient(p.inner, x)
    end

    _iso_run(parallel, n_chains) = begin
        empty!(_ISO_SEEN)
        probe = _IdProbe(_IsoGauss(exp.(range(-1.5, 1.5, 5))))
        rngs = [Xoshiro(1000 + i) for i in 1:n_chains]
        res = adaptive_warmup_mcmc(rngs, probe; n_draws=60, n_evaluations=120,
                                   stepsize_adaptation_limit=15, parallel)
        (; probe, res, seen = copy(_ISO_SEEN))
    end

    @testset "multi-chain adaptive: each chain gets its own lpdf" begin
        for parallel in (false, true)
            @testset "parallel=$parallel" begin
                n_chains = 3
                (; probe, res, seen) = _iso_run(parallel, n_chains)

                @test length(res) == n_chains
                @test !isempty(seen)
                # One distinct lpdf object per chain — never one shared object.
                @test length(unique(seen)) == n_chains
                # ...and none of them is the caller's, so the sampler cannot mutate it.
                @test objectid(probe) ∉ seen
            end
        end
    end

    @testset "multi-chain adaptive: a stateless lpdf is unaffected" begin
        # The common path must stay byte-identical: chain i of an n-chain run is the
        # same run as that chain on its own.
        cfg = (; n_draws=60, n_evaluations=120, stepsize_adaptation_limit=15)
        target = _IsoGauss(exp.(range(-1.5, 1.5, 5)))
        solo = [adaptive_warmup_mcmc(Xoshiro(2000 + i), target; cfg...) for i in 1:2]
        multi = adaptive_warmup_mcmc([Xoshiro(2001), Xoshiro(2002)], target; parallel=false, cfg...)
        for i in 1:2
            @test multi[i].posterior_position == solo[i].posterior_position
            @test multi[i].posterior_gradient == solo[i].posterior_gradient
        end
    end

    @testset "multi-chain adaptive: an explicit lpdfs vector is passed through as given" begin
        # Opting into sharing stays possible — `fill(lpdf, n)` on the array method is
        # the documented escape hatch, and it must NOT be silently copied.
        empty!(_ISO_SEEN)
        probe = _IdProbe(_IsoGauss(exp.(range(-1.5, 1.5, 5))))
        adaptive_warmup_mcmc([Xoshiro(3001), Xoshiro(3002)], fill(probe, 2);
                             n_draws=60, n_evaluations=120, stepsize_adaptation_limit=15,
                             parallel=false)
        @test unique(_ISO_SEEN) == [objectid(probe)]
    end
end
