# Reverse-mode (Enzyme) coverage for `ReparametrizedProblem`.
#
# WHY IT EXISTS: every other item in this suite pins `AutoForwardDiff()`, which is
# the one backend immune to the failure modes below. `dbde8d2` broke
# `Enzyme.Const` at the joint-transport site — the backend the
# `ReparametrizedProblem` docstring and both docs pages tell people to use — and
# landed fully green, because nothing committed here executed a reverse-mode
# backend. `aac6489` has since made that site exact and provable; this item is
# what made the break visible and is what would catch it coming back.
#
# WHY IT IS TAGGED: Enzyme's compile is heavy and the CI matrix is nine rows, so
# `--skip-tag=enzyme` keeps it out of all nine and one `ubuntu-latest` job runs
# `--tag=enzyme`. Before the TestItemRunner migration this was a whole separate
# environment (`web/src/test/enzyme/` with its own `Project.toml` and CI job); a
# tag is the same isolation without the second manifest to keep resolved. This is
# the arrangement StanBlocks uses for its `:bridgestan` items.
#
# THE REFERENCE IS A CLOSED FORM, not a second AD system: two AD backends
# agreeing does not establish that the objective is right, and the objective —
# `ljac(x) + dot(g_y, y(x))` with `g_y` frozen — is exactly what this item is for.
# The closed form is itself checked against central finite differences first, so
# it is not algebra asserted against itself.

@testitem "Enzyme reverse-mode" setup=[Targets, Determinism] tags=[:enzyme] begin
    using WarmupHMC, DifferentiationInterface, Enzyme, LogDensityProblems,
          Random, LinearAlgebra
    using WarmupHMC: IndexedReparametrization, Reparametrization, PartiallyCentered,
                     ReparametrizedProblem, reparametrizer, find_reparametrization!

    const K = 5
    spec(cs) = IndexedReparametrization([(i + 1) => Reparametrization(
        PartiallyCentered(1.0), PartiallyCentered(c), 0.0, x -> x[1] / 2)
        for (i, c) in enumerate(cs)])

    const CONST_ONLY = AutoEnzyme(; function_annotation = Enzyme.Const)
    const RUNTIME_ACTIVITY = AutoEnzyme(; mode = set_runtime_activity(Enzyme.Reverse),
                                          function_annotation = Enzyme.Const)
    const DUPLICATED = AutoEnzyme(; function_annotation = Enzyme.Duplicated)

    # --- the two references -------------------------------------------------
    #
    # CLOSED FORM for `spec(cs)` over `Funnel(K)`. `PartiallyCentered(c)` is the
    # parametrization in which the sampler coordinate `z` carries the model value
    # `v = location + scale^(1-c)·z`, so `c = 1` is fully centered (the coordinate
    # IS the value) and `c = 0` fully non-centered. Going from the target
    # parametrization `cᵢ` back to the source `1.0` therefore MULTIPLIES by
    # `scale^(1-cᵢ)` — with location 0 and `log_scale = x₁/2`:
    #
    #     ljacᵢ = (x₁/2)(1 - cᵢ)          yᵢ₊₁ = xᵢ₊₁ · exp(ljacᵢ)      y₁ = x₁
    #
    # so `L(x) = Σᵢ ljacᵢ + ld(y(x))` and, writing `gʸ = ∇ld(y)`,
    #
    #     ∂L/∂x₁   = Σᵢ (1-cᵢ)/2 + gʸ₁ + Σᵢ gʸᵢ₊₁ · yᵢ₊₁ · (1-cᵢ)/2
    #     ∂L/∂xᵢ₊₁ = gʸᵢ₊₁ · exp(ljacᵢ)
    #
    # This is the SAME identity the implementation computes by freezing `gʸ` and
    # taking one reverse pass over `x -> ljac(x) + dot(gʸ, y(x))` — which is the
    # point: it pins the objective, not just self-consistency between backends.
    #
    # The exponent's SIGN is the whole content of this reference, and it was
    # written inverted (`cᵢ - 1`) first. All three Enzyme backends still agreed
    # with EACH OTHER exactly — they all differentiate the same implementation —
    # so a reference built from any of them, or from a second AD system, would
    # have ratified the error. Only the finite-difference check below caught it.
    # Do not drop that check, and do not "simplify" it to another AD call.
    function analytic_value_and_gradient(cs, x)
        f = Funnel(K)
        ljac = [(x[1] / 2) * (1 - c) for c in cs]
        y = vcat(x[1], [x[i + 1] * exp(ljac[i]) for i in 1:K])
        ld, gy = LogDensityProblems.logdensity_and_gradient(f, y)
        g = similar(x)
        g[1] = sum((1 - c) / 2 for c in cs) + gy[1] +
               sum(gy[i + 1] * y[i + 1] * (1 - cs[i]) / 2 for i in 1:K)
        for i in 1:K
            g[i + 1] = gy[i + 1] * exp(ljac[i])
        end
        (sum(ljac) + ld, g)
    end

    # Central differences on the scalar logdensity — independent of any AD system
    # AND of the algebra above.
    function fd_gradient(rp, x; h = 1e-5)
        map(eachindex(x)) do i
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            (LogDensityProblems.logdensity(rp, xp) - LogDensityProblems.logdensity(rp, xm)) / 2h
        end
    end

    function halo(rp, n; seed = 4242)
        rng = Xoshiro(seed)
        X = Matrix{Float64}(undef, K + 1, n); G = similar(X)
        for j in 1:n
            v = 3randn(rng); s = exp(v / 2)
            X[:, j] = [v; s .* randn(rng, K)]
            G[:, j] = LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2]
        end
        X, G
    end

    # --- the reference is itself checked ------------------------------------
    #
    # Without this the machine-precision assertions below would be algebra
    # asserted against itself: a sign error in `analytic_value_and_gradient` and
    # in the implementation would cancel and read as agreement. FD is blunt
    # (~1e-11 error floor at h=1e-5) but it is derived from nothing but
    # `logdensity`, so the two cannot share a mistake.
    @testset "the closed-form reference agrees with finite differences" begin
        cs = fill(0.5, K)
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        rp = ReparametrizedProblem(spec(cs), Funnel(K), RUNTIME_ACTIVITY)
        lp, g = analytic_value_and_gradient(cs, x)
        @test lp ≈ LogDensityProblems.logdensity(rp, x)
        err = maximum(abs, g .- fd_gradient(rp, x))
        println("  reference  closed form vs central differences: ", err)
        @test err < 1e-6
    end

    # --- site 0: the documented landmine ------------------------------------
    #
    # A BARE `AutoEnzyme()` throws on any `ReparametrizedProblem` — the
    # differentiated objective closes over the reparametrizer and the frozen
    # `g_y`, which Enzyme cannot prove read-only.
    #
    # This is pinned FIRST because it is the one assertion here that can go stale
    # in the REASSURING direction. Every other test in this item fails loudly if
    # the behaviour it describes changes. This one would simply start passing for
    # the wrong reason: if a future Enzyme or DifferentiationInterface release
    # makes that closure provably read-only, the bare form begins working and the
    # `!!! warning` in the `ReparametrizedProblem` docstring becomes a false
    # claim in the public manual with nothing to catch it. When this test fails,
    # the fix is to update that docstring — not to delete this.
    @testset "a bare `AutoEnzyme()` still throws" begin
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        rp = ReparametrizedProblem(spec(fill(0.5, K)), Funnel(K), AutoEnzyme())
        err = try
            LogDensityProblems.logdensity_and_gradient(rp, x)
            nothing
        catch e
            e
        end
        @test err !== nothing
        # Not merely "something threw": that would let an unrelated MethodError
        # from a version bump keep this green while testing nothing at all. Pin
        # that it threw for the readonly/annotation reason the docstring names.
        msg = err === nothing ? "" : sprint(showerror, err)
        println("  site 0  bare AutoEnzyme() threw: ", nameof(typeof(err)))
        @test occursin("function_annotation", msg) || occursin("readonly", msg)
    end

    # --- site 1: the gradient hot path (`reparam_objective`) ----------------
    @testset "logdensity_and_gradient — every Enzyme variant, vs the closed form" begin
        cs = fill(0.5, K)
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        ref_lp, ref = analytic_value_and_gradient(cs, x)
        for (name, be) in (("Const", CONST_ONLY),
                           ("Const+set_runtime_activity", RUNTIME_ACTIVITY),
                           ("Duplicated", DUPLICATED))
            rp = ReparametrizedProblem(spec(cs), Funnel(K), be)
            lp, g = LogDensityProblems.logdensity_and_gradient(rp, x)
            @test lp ≈ LogDensityProblems.logdensity(rp, x)
            @test lp ≈ ref_lp
            err = maximum(abs, g .- ref)
            println("  site 1  ", rpad(name, 28), "max err vs closed form=", err)
            # Three orders tighter than the finite-difference tolerance this
            # replaced. At 1e-6 a small systematic error in the transport
            # identity — the exact thing worth catching — sits under the floor.
            @test err < 1e-12
        end
    end

    # --- site 1b: the assertion above is not vacuous -------------------------
    #
    # Two ways to get the objective subtly wrong, each of which the 1e-12
    # tolerance must reject. Both displacements are printed each run and are
    # orders of magnitude above 1e-12 — which is the point: the tolerance is not
    # doing the work, the identity is.
    @testset "a wrong objective would be caught" begin
        cs = fill(0.5, K)
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        _, ref = analytic_value_and_gradient(cs, x)

        f = Funnel(K)
        ljac = [(x[1] / 2) * (1 - c) for c in cs]
        y = vcat(x[1], [x[i + 1] * exp(ljac[i]) for i in 1:K])
        gy = LogDensityProblems.logdensity_and_gradient(f, y)[2]

        # (a) forget that `y` depends on x₁ through the log-scale closure
        no_args = copy(ref)
        no_args[1] -= sum(gy[i + 1] * y[i + 1] * (1 - cs[i]) / 2 for i in 1:K)
        # (b) forget the log-Jacobian term entirely
        no_ljac = copy(ref)
        no_ljac[1] -= sum((1 - c) / 2 for c in cs)

        println("  site 1b displacement  drop ∂args/∂x: ", maximum(abs, no_args .- ref),
                "   drop ∂ljac/∂x: ", maximum(abs, no_ljac .- ref))
        @test maximum(abs, no_args .- ref) > 1e-3
        @test maximum(abs, no_ljac .- ref) > 1e-3
    end

    # --- site 2: the joint halo transport (added by `dbde8d2`) --------------
    #
    # `transport_objective` is a SECOND `value_and_gradient` call site on the
    # same user-supplied backend. It used to be unprovable under a plain `Const`
    # — which is correct and sufficient for site 1 — and threw only at the first
    # restarting warm-up window, so a user following the docstring hit it partway
    # into a run rather than at setup. `aac6489` made it exact and statically
    # provable, so `CONST_ONLY` is asserted here as a plain `@test` alongside the
    # other two backends. Keeping that path exercised is the point: it is what
    # lets the docs drop the `set_runtime_activity` incantation.
    @testset "find_reparametrization! — the joint transport site" begin
        function run_transport(be)
            rp = ReparametrizedProblem(spec(fill(1.0, K)), Funnel(K), be)
            X, G = halo(rp, 200)
            pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict = false)
            find_reparametrization!(rp, X, G, pg)
            maximum(1:size(X, 2)) do j
                norm(LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2] .- G[:, j], Inf)
            end
        end

        for (name, be) in (("Const", CONST_ONLY),
                           ("Const+set_runtime_activity", RUNTIME_ACTIVITY),
                           ("Duplicated", DUPLICATED))
            err = run_transport(be)
            println("  site 2  ", rpad(name, 28), "max grad err=", err)
            # After the joint transport the halo gradient must equal a FRESH
            # evaluation at the transported point — that is the whole point of
            # the pass, and it is the property `optimize!` alone does not have.
            @test err < 1e-10
        end
    end

    # --- the FFI property ---------------------------------------------------
    #
    # `p.problem` is captured by the differentiated closure but never called
    # inside the traced region: `logdensity_and_gradient` runs BEFORE
    # `value_and_gradient` and `g_y` enters as a frozen `Vector{Float64}`. That
    # is what lets a BridgeStan target (which holds a raw `Ptr`) use a Julia
    # reverse-mode backend at all, so it is worth pinning rather than assuming.
    @testset "a captured raw Ptr is inert under AD" begin
        mutable struct PtrFunnel; k::Int; handle::Ptr{Cvoid}; end
        LogDensityProblems.dimension(f::PtrFunnel) = f.k + 1
        LogDensityProblems.capabilities(::Type{PtrFunnel}) = LogDensityProblems.LogDensityOrder{1}()
        LogDensityProblems.logdensity(f::PtrFunnel, x) = LogDensityProblems.logdensity(Funnel(f.k), x)
        LogDensityProblems.logdensity_and_gradient(f::PtrFunnel, x) =
            LogDensityProblems.logdensity_and_gradient(Funnel(f.k), x)

        sentinel = Ptr{Cvoid}(UInt(0xdeadbeef))
        pf = PtrFunnel(K, sentinel)
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        plain = LogDensityProblems.logdensity_and_gradient(
            ReparametrizedProblem(spec(fill(0.5, K)), Funnel(K), RUNTIME_ACTIVITY), x)
        held = LogDensityProblems.logdensity_and_gradient(
            ReparametrizedProblem(spec(fill(0.5, K)), pf, RUNTIME_ACTIVITY), x)
        @test held[1] == plain[1]
        @test held[2] == plain[2]
        @test pf.handle == sentinel
    end
end
