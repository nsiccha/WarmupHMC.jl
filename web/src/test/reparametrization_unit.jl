using WarmupHMC, Test, Random, LinearAlgebra, LogDensityProblems, Statistics
using InverseFunctions: inverse
using WarmupHMC: IndexedReparametrization, Reparametrization, PartiallyCentered,
                 ReparametrizedProblem, reparam, reparametrizer, optimize!,
                 reparam_sources, restore_reparam_sources!, find_reparametrization!

include(joinpath(@__DIR__, "ad_backend.jl"))
include(joinpath(@__DIR__, "targets.jl"))

# Unit / property guards for `src/Reparametrizations.jl` and its
# DifferentiationInterface-backed gradient path.
#
# These assert INVARIANTS rather than values, so they survive any legitimate
# change to what the adaptation happens to choose.

# Central-difference gradient of an arbitrary scalar function.
function fd_gradient(f, x; h=1e-6)
    map(eachindex(x)) do i
        step = h * max(1.0, abs(x[i]))
        xp = copy(x); xp[i] += step
        xm = copy(x); xm[i] -= step
        (f(xp) - f(xm)) / (2step)
    end
end

# The funnel's reparametrization, with an explicit per-coordinate source.
# Coordinate 1 (`v`) carries the log-scale and is itself never reparametrized,
# so the `args` closures read a coordinate the transform leaves alone.
funnel_ir(sources) = IndexedReparametrization([
    (i + 1) => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                                 0.0, x -> x[1] / 2)
    for (i, c) in enumerate(sources)
])
funnel_problem(sources) =
    ReparametrizedProblem(funnel_ir(sources), Funnel(length(sources)), AutoForwardDiff())

# A three-level funnel: `v` (log-scale) → `mu` (location) → `theta[1:k]`.
#
# The funnel above cannot express the case that matters below, because its `loc`
# is the constant `0.` and its log-scale coordinate is never reparametrized. Here
# `mu` is a real coordinate that the θ blocks read as their `loc`, so it can be
# BOTH a closure argument and a reparametrization target — the coupled spec.
#
#   v ~ Normal(0, 3),  mu ~ Normal(0, exp(v/2)),  theta[i] ~ Normal(mu, exp(v/2))
struct NestedFunnel
    k::Int
end
LogDensityProblems.dimension(m::NestedFunnel) = m.k + 2
LogDensityProblems.capabilities(::Type{NestedFunnel}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(m::NestedFunnel, y)
    v = y[1]; mu = y[2]; th = @view y[3:end]; e = exp(-v)
    -0.5 * v^2 / 9 - 0.5 * mu^2 * e - v / 2 - sum(t -> 0.5 * (t - mu)^2 * e + v / 2, th)
end
function LogDensityProblems.logdensity_and_gradient(m::NestedFunnel, y)
    v = y[1]; mu = y[2]; th = @view y[3:end]; e = exp(-v)
    g = similar(y)
    g[1] = -v / 9 + 0.5 * mu^2 * e - 0.5 + 0.5 * sum(t -> (t - mu)^2, th) * e - m.k / 2
    g[2] = -mu * e + sum(t -> t - mu, th) * e
    g[3:end] .= .-(th .- mu) .* e
    (LogDensityProblems.logdensity(m, y), g)
end

@testset "Reparametrizations" begin

    @testset "PartiallyCentered round-trip and inverse" begin
        rng = Xoshiro(1)
        for _ in 1:20
            t, s = rand(rng), rand(rng)
            loc, log_scale = 3randn(rng), randn(rng)
            x = 2randn(rng)
            r = Reparametrization(PartiallyCentered(t), PartiallyCentered(s), loc, log_scale)

            ljac, y = reparam(r, x, nothing)          # args are constants here
            # The documented map: y = t*loc + (x - s*loc)*exp(log_scale*(t - s)),
            # with log-Jacobian log_scale*(t - s).
            @test ljac ≈ log_scale * (t - s)
            @test y ≈ t * loc + (x - s * loc) * exp(log_scale * (t - s))

            # `inverse` swaps target and source, so it must undo the map exactly
            # and negate the log-Jacobian.
            ljac_back, x_back = reparam(inverse(r), y, nothing)
            @test x_back ≈ x
            @test ljac_back ≈ -ljac
            @test ljac + ljac_back ≈ 0 atol = 1e-12
        end
    end

    @testset "PartiallyCentered with source == target is the identity" begin
        rng = Xoshiro(2)
        for _ in 1:10
            c = rand(rng)
            r = Reparametrization(PartiallyCentered(c), PartiallyCentered(c), 2randn(rng), randn(rng))
            x = randn(rng)
            ljac, y = reparam(r, x, nothing)
            @test ljac == 0
            @test y ≈ x
        end
    end

    @testset "the three-argument form transports the gradient consistently" begin
        # `reparam(r, x, g, …)` must return dx/dy applied to g, i.e. the chain
        # rule for the coordinate being mapped.
        rng = Xoshiro(3)
        for _ in 1:20
            t, s = rand(rng), rand(rng)
            loc, log_scale = 3randn(rng), randn(rng)
            x, g = 2randn(rng), randn(rng)
            r = Reparametrization(PartiallyCentered(t), PartiallyCentered(s), loc, log_scale)
            ljac2, y2 = reparam(r, x, nothing)
            ljac3, y3, g3 = reparam(r, x, g, nothing)
            @test (ljac3, y3) == (ljac2, y2)
            # dy/dx = exp(ljac), so the pullback of g is g / exp(ljac).
            @test g3 ≈ g / exp(ljac3)
        end
    end

    @testset "IndexedReparametrization inverse round-trips the full vector" begin
        rng = Xoshiro(4)
        ir = funnel_ir([0.2, 0.8, 0.5])
        for _ in 1:20
            x = randn(rng, 4)
            ljac, y = ir(x)
            ljac_back, x_back = inverse(ir)(y)
            @test x_back ≈ x
            @test ljac_back ≈ -ljac
            # Untouched coordinates must pass through byte-identically.
            @test y[1] == x[1]
        end
    end

    @testset "the log-Jacobian is the true log|det ∂y/∂x|" begin
        # The correction added to the log density must be the actual Jacobian of
        # the map, or every reparametrized posterior is silently the wrong
        # distribution.
        rng = Xoshiro(5)
        ir = funnel_ir([0.1, 0.9, 0.4])
        for _ in 1:10
            x = randn(rng, 4)
            ljac, _ = ir(x)
            J = ForwardDiff.jacobian(z -> ir(z)[2], x)
            @test ljac ≈ log(abs(det(J)))
        end
    end

    @testset "an empty reparametrizer is an exact no-op" begin
        # Trap 1: `reparametrizer(::Any)` is the empty `IndexedReparametrization`,
        # so the whole machinery must vanish on a plain lpdf.
        ir = IndexedReparametrization([])
        x = randn(Xoshiro(6), 5)
        ljac, y = ir(x)
        @test ljac == 0
        @test y == x
        @test isempty(reparametrizer(Funnel(4)).pairs)
        @test isempty(reparam_sources(Funnel(4)))
        @test restore_reparam_sources!(Funnel(4), []) isa Funnel
    end

    @testset "ReparametrizedProblem gradient path (DifferentiationInterface)" begin
        rng = Xoshiro(7)
        for sources in ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.3, 0.7, 0.5])
            rp = funnel_problem(sources)
            @test LogDensityProblems.dimension(rp) == 4
            for _ in 1:10
                x = 0.6 .* randn(rng, 4)
                lp, g = LogDensityProblems.logdensity_and_gradient(rp, x)
                # The value must agree with `logdensity`, which composes
                # independently of the gradient path.
                @test lp ≈ LogDensityProblems.logdensity(rp, x)
                # And the gradient must be the gradient OF THAT value — the
                # extension differentiates only through the transform and reuses
                # the inner problem's native gradient, so this is the check that
                # the chain rule is assembled correctly.
                ref = fd_gradient(z -> LogDensityProblems.logdensity(rp, z), x)
                @test g ≈ ref rtol = 1e-5
                # Cross-check against full AD through the composed density.
                @test g ≈ ForwardDiff.gradient(z -> LogDensityProblems.logdensity(rp, z), x) rtol = 1e-9
            end
        end
    end

    @testset "a centered ReparametrizedProblem reproduces its inner problem" begin
        # source == target == 1.0 makes the transform the identity, so the
        # wrapper must be indistinguishable from the bare model. If this fails,
        # every "reparametrized" benchmark is measuring a different posterior.
        rng = Xoshiro(8)
        rp = funnel_problem([1.0, 1.0, 1.0])
        inner = Funnel(3)
        for _ in 1:10
            x = randn(rng, 4)
            lp_in, g_in = LogDensityProblems.logdensity_and_gradient(inner, x)
            lp_rp, g_rp = LogDensityProblems.logdensity_and_gradient(rp, x)
            @test lp_rp ≈ lp_in
            @test g_rp ≈ g_in
        end
    end

    @testset "reparam_sources / restore_reparam_sources! round-trip" begin
        rp = funnel_problem([0.2, 0.6, 0.9])
        captured = reparam_sources(rp)
        @test [idx for (idx, _) in captured] == [2, 3, 4]
        @test [s.c for (_, s) in captured] == [0.2, 0.6, 0.9]

        # A freshly built problem always starts from the sources it was
        # constructed with; restore must overwrite them with the captured ones.
        fresh = funnel_problem([1.0, 1.0, 1.0])
        @test [s.c for (_, s) in reparam_sources(fresh)] == [1.0, 1.0, 1.0]
        restore_reparam_sources!(fresh, captured)
        @test [s.c for (_, s) in reparam_sources(fresh)] == [0.2, 0.6, 0.9]
        # Targets and the arg closures must survive untouched.
        for (a, b) in zip(reparametrizer(fresh).pairs, reparametrizer(rp).pairs)
            @test first(a) == first(b)
            @test last(a).target.c == last(b).target.c
        end
        # And restoring makes the two problems evaluate identically.
        x = randn(Xoshiro(9), 4)
        @test LogDensityProblems.logdensity(fresh, x) ≈ LogDensityProblems.logdensity(rp, x)

        # Restore is positional: it zips `ir.pairs` with `sources` in order. Pin
        # that, because it is what makes coordinate ORDER load-bearing on resume.
        @test length(reparametrizer(fresh).pairs) == length(captured)
    end

    @testset "optimize! transports the halo into the new source coordinates" begin
        # The marginal search re-expresses each fitted coordinate as it goes.
        # Its local gradient update is intentionally only diagonal; the exact
        # full-vector correction happens once in `find_reparametrization!`,
        # after every marginal source has been selected.
        rng = Xoshiro(10)
        start_sources = [1.0, 1.0, 1.0]
        rp = funnel_problem(start_sources)
        ir = reparametrizer(rp)

        # A halo of real states with their real gradients, in source coordinates.
        n = 400
        X = Matrix{Float64}(undef, 4, n)
        G = Matrix{Float64}(undef, 4, n)
        for j in 1:n
            v = 3randn(rng)
            x = [v; exp(v / 2) .* randn(rng, 3)]
            X[:, j] = x
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, x)[2]
        end
        X0, G0 = copy(X), copy(G)

        optimize!(ir, X, G)
        new_sources = [s.c for (_, s) in reparam_sources(rp)]
        println("  optimize! moved sources $(start_sources) -> $(new_sources)")
        @test all(0 .<= new_sources .<= 1)

        # (1) POSITION transport: the model-coordinate point must be invariant.
        # `ir_old(x_old)` and `ir_new(x_new)` are the same y, or the halo now
        # describes different states than the sampler visited.
        ir_old = funnel_ir(start_sources)
        for j in 1:n
            @test ir(X[:, j])[2] ≈ ir_old(X0[:, j])[2] rtol = 1e-10
        end
        # It genuinely moved (guards against a vacuous pass when nothing changed).
        @test new_sources != start_sources
        @test !(X ≈ X0)

        # (2) GRADIENT transport: the transported gradient rows must equal the
        # gradient of the NEW source parametrization at the transported point.
        rp_new = funnel_problem(new_sources)
        idxs = [idx for (idx, _) in reparametrizer(rp).pairs]
        max_err = 0.0
        for j in 1:n
            g_true = LogDensityProblems.logdensity_and_gradient(rp_new, X[:, j])[2]
            for i in idxs
                max_err = max(max_err, abs(G[i, j] - g_true[i]) / max(1.0, abs(g_true[i])))
            end
        end
        println("  max relative gradient-transport error on reparametrized rows: $max_err")
        @test max_err < 1e-8
    end

    @testset "find_reparametrization! jointly transports the full halo" begin
        # The marginal-only updates inside `optimize!` are intentional. Once all
        # sources settle, the sampler seam must rematerialize position and
        # gradient together before the linear metric reads the pool. Row 1 is
        # the key guard: it is not reparametrized, but controls every block's
        # log-scale and therefore receives cross-coordinate pullback terms.
        rng = Xoshiro(11)
        start_sources = [1.0, 1.0, 1.0]
        rp = funnel_problem(start_sources)
        n = 400
        X = Matrix{Float64}(undef, 4, n)
        G = Matrix{Float64}(undef, 4, n)
        for j in 1:n
            v = 3randn(rng)
            x = [v; exp(v / 2) .* randn(rng, 3)]
            X[:, j] = x
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, x)[2]
        end
        X0 = copy(X)
        ir_old = funnel_ir(start_sources)
        pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict=false)
        counted = WarmupHMC.CountingPosterior(rp)

        find_reparametrization!(counted, X, G, pg)
        new_sources = [s.c for (_, s) in reparam_sources(rp)]
        @test new_sources != start_sources

        max_err = 0.0
        for j in 1:n
            @test reparametrizer(rp)(X[:, j])[2] ≈ ir_old(X0[:, j])[2] rtol = 1e-10
            g_true = LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2]
            max_err = max(max_err, maximum(abs.(G[:, j] .- g_true)))
            @test G[:, j] ≈ g_true rtol = 1e-10 atol = 1e-10
        end
        println("  full-vector max absolute gradient-transport error: $max_err")

        # The pool transport itself performs no model evaluations. The sole
        # counted call is the pre-existing refresh of the live HMC state.
        @test counted.count[] == 1
    end

    @testset "joint transport gives the linear stage the exact funnel scale" begin
        # `adaptive_warmup_mcmc.jl` fits the linear metric immediately after the
        # joint pass, so this checks the corrected pool in the units the sampler
        # actually consumes.
        #
        # The reference value here is exact, not measured. `update_loss!` for a
        # `Diagonal` sets `t[i,i] = sqrt(std(pᵢ)/std(gᵢ))`; the funnel fits to
        # fully non-centered (`c = 0`), and in THAT frame the target is exactly
        # `v ~ Normal(0, 3)` with `∂logp/∂v = -v/9`. The ratio is therefore 9
        # regardless of which draws we happened to take — sampling noise cancels
        # between numerator and denominator — so the correct `t[1,1]` is 3.
        rng = Xoshiro(11)
        rp = funnel_problem([1.0, 1.0, 1.0])
        n = 400
        X = Matrix{Float64}(undef, 4, n)
        G = Matrix{Float64}(undef, 4, n)
        for j in 1:n
            v = 3randn(rng)
            x = [v; exp(v / 2) .* randn(rng, 3)]
            X[:, j] = x
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, x)[2]
        end
        pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict=false)
        find_reparametrization!(rp, X, G, pg)
        @test all(iszero(s.c) for (_, s) in reparam_sources(rp))

        scale = Diagonal(ones(4))
        WarmupHMC.update_loss!(scale, X, G)
        println("  diagonal t[1,1] after joint transport: $(scale[1,1])")
        @test scale[1, 1] ≈ 3.0 rtol = 1e-12
    end

    @testset "joint transport corrects `loc` closure rows too" begin
        # The full pass must cover more than log-scale rows. Coordinate 2 is the
        # shared `loc` of four blocks and is not itself reparametrized, so only
        # the joint pullback supplies its chain-rule term.
        rng = Xoshiro(4242)
        k = 4
        ir_of(cs) = IndexedReparametrization([
            (i + 2) => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                                         x -> x[2], x -> x[1] / 2)
            for (i, c) in enumerate(cs)
        ])
        prob_of(cs) = ReparametrizedProblem(ir_of(cs), NestedFunnel(k), AutoForwardDiff())

        rp = prob_of(ones(k))
        n = 400
        X = Matrix{Float64}(undef, k + 2, n)
        G = Matrix{Float64}(undef, k + 2, n)
        for j in 1:n
            v = 3randn(rng); s = exp(v / 2); mu = s * randn(rng)
            X[:, j] = [v; mu; mu .+ s .* randn(rng, k)]
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2]
        end
        pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict=false)
        find_reparametrization!(rp, X, G, pg)
        gerr = [maximum(1:n) do j
                    g_true = LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2]
                    abs(G[r, j] - g_true[r])
                end for r in 1:(k + 2)]
        println("  gradient error per row (rows 1,2 are the closure coordinates): ",
                round.(gerr, sigdigits=4))

        @test maximum(gerr) < 1e-8
    end

    @testset "coupled `loc`: a reparametrized coordinate as another block's loc" begin
        # The one cross-coordinate shape the joint pass does NOT handle: a
        # coordinate that is simultaneously a transform target and another
        # block's `loc` argument. The testset directly above is the uncoupled
        # control — coordinate 2 is the shared `loc` but is not in `pairs`, and
        # there the joint pass is exact. Here it is both.
        #
        # This measures the SHIPPED seam, `find_reparametrization!`, and not the
        # marginal `optimize!` search it wraps. That distinction is load-bearing:
        # while this testset called `optimize!` it reported the marginal number,
        # stayed Broken for a right-looking reason, and could not see that
        # `ba6b4f01` made the coupled case WORSE. Measured on `ba6b4f01` — same
        # harness, same seed, the seam being the only difference:
        #
        #   start          marginal (`optimize!`)     joint (`find_reparametrization!`)
        #   c = 1 (all)    pos 1.4e-14 (exact)   ->   pos 1.9e+02   grad 5.6e+03
        #   c = 0.5 (all)  pos 1.1e+02           ->   pos 4.2e+02   grad 1.0e+03
        #
        # Read the first row: from a fully-centered start — where every shipped
        # spec begins — the marginal search was exact and the joint pass breaks
        # it. The comment that used to stand here called that regime "correct,
        # measured exact to 1.4e-14" and on that basis predicted this pin would
        # promote on the joint fix. It did the opposite. Both regimes are broken
        # now, so the old first-window/later-window split describes nothing.
        #
        # LATENT, not a live bug: every spec shipped in this repo is uncoupled
        # (the `args` coordinates are disjoint from the `idx` set in all nine
        # families), so nothing reaches this today. But `accel_gp` reads `x[46]`
        # with its `idx` starting at 47 — one off-by-one away — and after
        # `ba6b4f01` that off-by-one costs the FIRST window rather than the
        # second.
        #
        # All four pins below are `@test_broken` deliberately. Any of them
        # promoting is the signal that the coupled composition got fixed; the
        # position pair and the gradient pair can move independently.
        k = 4
        coupled_ir(cs) = IndexedReparametrization(vcat(
            [2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(cs[1]),
                                    0.0, x -> x[1] / 2)],
            [(i + 2) => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(c),
                                          x -> x[2], x -> x[1] / 2)
             for (i, c) in enumerate(cs[2:end])]))
        coupled_problem(cs) =
            ReparametrizedProblem(coupled_ir(cs), NestedFunnel(k), AutoForwardDiff())

        # Both invariants the sampler needs out of the seam: the MODEL point each
        # column denotes must not move, and the stored gradient must be the one
        # the target actually returns at the new coordinates.
        function coupled_transport_error(start)
            rng = Xoshiro(4242)
            rp = coupled_problem(start)
            ir = reparametrizer(rp)
            ir_old = coupled_ir(start)
            n = 400
            X = Matrix{Float64}(undef, k + 2, n)
            G = Matrix{Float64}(undef, k + 2, n)
            for j in 1:n
                v = 3randn(rng); s = exp(v / 2); mu = s * randn(rng)
                X[:, j] = [v; mu; mu .+ s .* randn(rng, k)]
                G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2]
            end
            X0 = copy(X)
            pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict=false)
            find_reparametrization!(rp, X, G, pg)
            moved = maximum(abs, view(X, 2, :) .- view(X0, 2, :))
            perr = maximum(3:(k + 2)) do r
                maximum(j -> abs(ir(X[:, j])[2][r] - ir_old(X0[:, j])[2][r]), 1:n)
            end
            gerr = maximum(1:n) do j
                maximum(abs, G[:, j] .-
                        LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2])
            end
            (; perr, gerr, moved)
        end

        centered = coupled_transport_error(fill(1.0, k + 1))
        println("  coupled, c_old == c_target: joint θ-position error = $(centered.perr), ",
                "gradient error = $(centered.gerr) (loc coordinate moved by $(centered.moved))")
        # Not vacuous: the `loc` coordinate really does move underneath the θ
        # blocks. This one passed as `@test` while the seam was `optimize!`.
        @test centered.moved > 1.0
        @test_broken centered.perr < 1e-10
        @test_broken centered.gerr < 1e-8

        shifted = coupled_transport_error(fill(0.5, k + 1))
        println("  coupled, c_old != c_target: joint θ-position error = $(shifted.perr), ",
                "gradient error = $(shifted.gerr) (loc coordinate moved by $(shifted.moved))")
        @test shifted.moved > 1.0
        @test_broken shifted.perr < 1e-10
        @test_broken shifted.gerr < 1e-8
    end

    @testset "find_reparametrization! is a no-op for a plain lpdf" begin
        # Trap 1 again, at the sampler seam: `nonlinear_adapt=true` on a bare
        # target must change nothing at all.
        rng = Xoshiro(12)
        lpdf = Funnel(3)
        X = randn(rng, 4, 50)
        G = randn(rng, 4, 50)
        X0, G0 = copy(X), copy(G)
        pg = WarmupHMC.DynamicHMC.evaluate_ℓ(lpdf, randn(rng, 4); strict=false)
        out = find_reparametrization!(lpdf, X, G, pg)
        @test out === pg          # returns the position untouched
        @test X == X0
        @test G == G0
    end
end
