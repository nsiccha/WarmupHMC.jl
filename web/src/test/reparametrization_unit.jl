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
        # THE central claim of the adaptation step: after `optimize!`, the halo
        # matrices must describe the SAME points, re-expressed in the newly
        # chosen source parametrization — for position AND gradient — because
        # the linear metric `argmin` immediately refits on them.
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

    @testset "gradient transport is diagonal-only: the log-scale row goes stale" begin
        # Known rough edge (established, not fixed here): `optimize!` rewrites
        # only the rows it reparametrizes. Row 1 carries the log-scale that every
        # transform reads, so its gradient picks up a chain-rule term that is
        # never applied. The linear metric `argmin` then fits that row on stale
        # values.
        rng = Xoshiro(11)
        rp = funnel_problem([1.0, 1.0, 1.0])
        ir = reparametrizer(rp)
        n = 400
        X = Matrix{Float64}(undef, 4, n)
        G = Matrix{Float64}(undef, 4, n)
        for j in 1:n
            v = 3randn(rng)
            x = [v; exp(v / 2) .* randn(rng, 3)]
            X[:, j] = x
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, x)[2]
        end
        optimize!(ir, X, G)
        rp_new = funnel_problem([s.c for (_, s) in reparam_sources(rp)])
        err = maximum(1:n) do j
            g_true = LogDensityProblems.logdensity_and_gradient(rp_new, X[:, j])[2]
            abs(G[1, j] - g_true[1])
        end
        println("  log-scale row (1) max absolute gradient error after transport: $err")
        @test_broken err < 1e-8
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
