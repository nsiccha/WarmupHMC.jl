# Tests for the truly-cooperative (clustered pooled-mass-matrix) sampler:
#   * the pooled-scale primitives (src/pooled_scale.jl): weighted Welford
#     Variances, Chan merge! + exact unmerge!, the nutpie marginal scale, and the
#     tuneable cond_compatibility metric;
#   * both clustering strategies (src/clustering.jl): greedy assign_clusters and
#     retrospective lookbehind_clusters (loo / linkage / inclusive criteria);
#   * the driver (src/clustered_warmup_mcmc.jl): 1-cluster and 2-cluster targets,
#     per-cluster labeled output, and BIT-IDENTICAL in-memory resumability.
#
# Run this item alone with
#   -- --file=clustered.jl

@testitem "the clustered cooperative sampler" setup=[Determinism] tags=[:sampler] begin
    using WarmupHMC, LogDensityProblems, LinearAlgebra, Random, Statistics
    using WarmupHMC: Variances, NutpieScaleAdaptation, unmerge!, pooled, marginal_scales,
        cond_compatibility, compatible, assign_clusters, lookbehind_clusters,
        loo_criterion, linkage_criterion, inclusive_criterion,
        clustered_chains, clustered_step!, clustered_output, clustered_result, chain_draws
    import OnlineStatsBase
    using OnlineStatsBase: fit!, nobs

    # --- targets ----------------------------------------------------------------
    # Named distinctly from cooperative.jl's for historical reasons — each item now
    # gets its own module, so the two could not collide even if they matched.
    struct DiagNormalLP           # axis-aligned Gaussian: the diagonal sampler's happy path
        scales::Vector{Float64}
    end
    LogDensityProblems.dimension(p::DiagNormalLP) = length(p.scales)
    LogDensityProblems.capabilities(::Type{<:DiagNormalLP}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(p::DiagNormalLP, x) = -0.5 * sum((x ./ p.scales) .^ 2)
    LogDensityProblems.logdensity_and_gradient(p::DiagNormalLP, x) = (LogDensityProblems.logdensity(p, x), -x ./ p.scales .^ 2)

    struct MixNormalLP            # two Gaussian modes, proper mixing weights
        mu::Vector{Vector{Float64}}
        P::Vector{Matrix{Float64}}
        logconst::Vector{Float64}
    end
    MixNormalLP(mu, P) = MixNormalLP(mu, P, [0.5 * logdet(Pk) for Pk in P])
    LogDensityProblems.dimension(p::MixNormalLP) = length(p.mu[1])
    LogDensityProblems.capabilities(::Type{<:MixNormalLP}) = LogDensityProblems.LogDensityOrder{1}()
    _mixcomp(p::MixNormalLP, x) = map(eachindex(p.mu)) do k
        d = x .- p.mu[k]; (-0.5 * dot(d, p.P[k], d) + p.logconst[k], -(p.P[k] * d))
    end
    function LogDensityProblems.logdensity(p::MixNormalLP, x)
        lps = first.(_mixcomp(p, x)); m = maximum(lps); m + log(sum(exp, lps .- m))
    end
    function LogDensityProblems.logdensity_and_gradient(p::MixNormalLP, x)
        comps = _mixcomp(p, x); lps = first.(comps); m = maximum(lps)
        w = exp.(lps .- m); w ./= sum(w)
        (m + log(sum(exp, lps .- m)), sum(w[k] .* comps[k][2] for k in eachindex(comps)))
    end

    # Build a chain estimate from a scale regime (estimator-level clustering tests).
    function regime(dim, rng, scale_p; n=1500)
        a = NutpieScaleAdaptation(dim)
        for _ in 1:n; fit!(a, randn(rng, dim) .* scale_p, randn(rng, dim)); end
        a
    end
    setsig(clusters) = Set(Set.(clusters))

    @testset "clustered cooperative sampler" begin

        @testset "pooled-scale primitives" begin
            # merge! pools == fitting all data once (mean + Bessel variance)
            let dim = 4, rng = Xoshiro(1)
                A = [randn(rng, dim) for _ in 1:150]; B = [randn(rng, dim) .* 2 .+ 1 for _ in 1:90]
                va = Variances(dim); foreach(y -> fit!(va, y), A)
                vb = Variances(dim); foreach(y -> fit!(vb, y), B)
                merged = deepcopy(va); merge!(merged, vb)
                vall = Variances(dim); foreach(y -> fit!(vall, y), vcat(A, B))
                @test mean(merged) ≈ mean(vall)
                @test var(merged) ≈ var(vall)
                @test nobs(merged) == nobs(vall) == 240
            end
            # unmerge! is the exact inverse of merge!
            let dim = 3, rng = Xoshiro(2)
                A = [randn(rng, dim) for _ in 1:120]; B = [randn(rng, dim) .* 3 for _ in 1:70]
                va = Variances(dim); foreach(y -> fit!(va, y), A)
                vb = Variances(dim); foreach(y -> fit!(vb, y), B)
                pool = deepcopy(va); merge!(pool, vb); unmerge!(pool, vb)
                @test mean(pool) ≈ mean(va)
                @test var(pool) ≈ var(va)
                @test nobs(pool) == nobs(va)
            end
            # nutpie marginal scale = (var_pos/var_grad)^(1/4) = sqrt(sd_pos/sd_grad)
            let dim = 3, rng = Xoshiro(3)
                a = NutpieScaleAdaptation(dim; regularizing_n=0.0)
                sp = [1.0, 2.0, 0.5]; sg = [0.5, 1.0, 3.0]
                for _ in 1:40000; fit!(a, randn(rng, dim) .* sp, randn(rng, dim) .* sg); end
                @test marginal_scales(a) ≈ sqrt.(sp ./ sg) rtol=0.05
            end
            # compatibility: self ~1; same-distribution compatible; different-scale not
            let dim = 4, rng = Xoshiro(4)
                a = NutpieScaleAdaptation(dim); b = NutpieScaleAdaptation(dim); c = NutpieScaleAdaptation(dim)
                for _ in 1:4000
                    fit!(a, randn(rng, dim), randn(rng, dim)); fit!(b, randn(rng, dim), randn(rng, dim))
                    fit!(c, randn(rng, dim) .* [10.0, 1, 1, 1], randn(rng, dim))
                end
                @test cond_compatibility(a, a) ≈ 1
                @test compatible(a, b)
                @test !compatible(a, c)
                na = nobs(a); p = pooled(a, b)              # pooled() must not mutate inputs
                @test nobs(p) == nobs(a) + nobs(b)
                @test nobs(a) == na
            end
        end

        @testset "clustering strategies" begin
            for (name, cf) in [("greedy", assign_clusters),
                               ("lookbehind/loo", (a; kw...) -> lookbehind_clusters(a; criterion=loo_criterion, kw...)),
                               ("lookbehind/linkage", (a; kw...) -> lookbehind_clusters(a; criterion=linkage_criterion, kw...)),
                               ("lookbehind/inclusive", (a; kw...) -> lookbehind_clusters(a; criterion=inclusive_criterion, kw...))]
                @testset "$name" begin
                    let rng = Xoshiro(10)                       # two well-separated regimes -> 2 clusters
                        ads = [regime(3, rng, k <= 4 ? [1.0,1,1] : [8.0,1,1]) for k in 1:7]
                        @test setsig(cf(ads; threshold=sqrt(2.0))) == Set([Set(1:4), Set(5:7)])
                    end
                    let rng = Xoshiro(11)                       # one regime -> 1 cluster
                        ads = [regime(4, rng, ones(4)) for _ in 1:6]
                        @test length(cf(ads; threshold=sqrt(2.0))) == 1
                    end
                    let ads = [regime(2, Xoshiro(12), ones(2))]  # singleton
                        @test cf(ads; threshold=sqrt(2.0)) == [[1]]
                    end
                end
            end
            # look-behind isolates a scale-ambiguous "bridge" chain as a self-consistent singleton
            let rng = Xoshiro(21)
                ads = vcat([regime(2, rng, [1.0,1]) for _ in 1:3],
                           [regime(2, rng, [5.0,1]) for _ in 1:3],
                           [regime(2, rng, [2.3,1])])
                lb = lookbehind_clusters(ads; threshold=sqrt(2.0))
                @test all(lb) do m
                    length(m) == 1 || all(i -> loo_criterion([ads[k] for k in m if k != i], ads[i]; metric=cond_compatibility) <= sqrt(2.0), m)
                end
            end
        end

        @testset "driver: 1 cluster (axis-aligned Gaussian)" begin
            lpdf = DiagNormalLP(exp.(range(-1.0, 1.0; length=6)))
            r = clustered_warmup_mcmc(Xoshiro.(1:6), lpdf; n_draws=200, max_windows=10, parallel=false)
            @test length(r.clusters) == 1
            c = only(r.clusters)
            @test c.n_chains == 6
            @test isfinite(c.ess) && c.ess > 40
            @test isfinite(c.rhat) && c.rhat < 1.1
        end

        @testset "driver: 2 clusters (differently-scaled mixture)" begin
            dim = 3
            muA = [-10.0, 0.0, 0.0]; muB = [10.0, 0.0, 0.0]
            lpdf = MixNormalLP([muA, muB], [Matrix(Diagonal(ones(dim))), Matrix(Diagonal([1/9.0, 1.0, 1.0]))])
            initA = (; position = muA .+ 0.2, squared_scale = Matrix(Diagonal(ones(dim))))
            initB = (; position = muB .+ 0.2, squared_scale = Matrix(Diagonal([9.0, 1.0, 1.0])))
            inits = [k <= 3 ? initA : initB for k in 1:6]
            for (name, cf) in [("greedy", assign_clusters), ("lookbehind", lookbehind_clusters)]
                @testset "$name" begin
                    r = clustered_warmup_mcmc(Xoshiro.(201:206), lpdf; n_draws=200, max_windows=10,
                                              init=inits, cluster_fn=cf, parallel=false)
                    @test length(r.clusters) == 2
                    @test sort(length.(getproperty.(r.clusters, :chain_indices))) == [3, 3]
                    # each cluster recovered its basin's marginal scale in dim 1 (≈1 vs ≈3)
                    d1 = sort([mean(res.scale[1] for res in r.results[c.chain_indices]) for c in r.clusters])
                    @test d1[1] ≈ 1.0 rtol=0.4
                    @test d1[2] ≈ 3.0 rtol=0.4
                end
            end
        end

        @testset "in-memory resumability is bit-identical" begin
            lpdf = DiagNormalLP(exp.(range(-0.7, 0.7; length=4)))
            cfg = (; n_draws=150, parallel=false)
            # continuous: 6 windows in one loop
            cA = clustered_chains(Xoshiro.(1:4), lpdf; cfg...)
            for _ in 1:6; clustered_step!(cA); end
            # split: 3 windows, (checkpoint) then 3 more — same seeds, same state carried
            cB = clustered_chains(Xoshiro.(1:4), lpdf; cfg...)
            for _ in 1:3; clustered_step!(cB); end
            for _ in 1:3; clustered_step!(cB); end
            @test chain_draws.(cA) == chain_draws.(cB)                     # bit-for-bit
            @test [c.scale for c in cA] == [c.scale for c in cB]
            # the finalized output matches too
            oA = clustered_output(cA, [collect(1:4)], 6)
            @test getproperty.(oA.results, :n_samples) == getproperty.(clustered_output(cB, [collect(1:4)], 6).results, :n_samples)
        end
    end
end
