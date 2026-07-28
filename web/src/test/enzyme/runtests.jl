# Reverse-mode (Enzyme) coverage for `ReparametrizedProblem`.
#
# WHY THIS EXISTS AS A SEPARATE ENVIRONMENT: see the header of `Project.toml`.
# Path note: `test/` is a symlink to `web/src/test/`; CI uses the real path.
# WHY IT EXISTS AT ALL: every other environment in this repo pins
# `AutoForwardDiff()`, which is the one backend immune to the two failure modes
# below. `dbde8d2` broke `Enzyme.Const` — the backend the `ReparametrizedProblem`
# docstring and both docs pages tell people to use — and landed fully green,
# because nothing committed here executed a reverse-mode backend. This file is
# that missing gate.
#
# The reference for every gradient assertion is a CENTRAL FINITE DIFFERENCE of
# `LogDensityProblems.logdensity(rp, x)`, deliberately not a second AD system:
# two AD backends agreeing does not establish that the objective is right, and
# the objective (`ljac(x) + dot(g_y, y(x))` with `g_y` frozen) is what these
# tests are for. There is no ForwardDiff in this environment on purpose.

using Test, WarmupHMC, DifferentiationInterface, Enzyme, LogDensityProblems,
      Random, LinearAlgebra, DynamicHMC
using WarmupHMC: IndexedReparametrization, Reparametrization, PartiallyCentered,
                 ReparametrizedProblem, reparametrizer, find_reparametrization!

struct Funnel; k::Int; end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]
    -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
end
LogDensityProblems.logdensity_and_gradient(f::Funnel, x) = begin
    v = x[1]; xs = @view x[2:end]; g = similar(x)
    g[1] = -v / 9 + 0.5 * exp(-v) * sum(abs2, xs) - f.k / 2
    g[2:end] .= .-xs .* exp(-v)
    (LogDensityProblems.logdensity(f, x), g)
end

const K = 5
spec(cs) = IndexedReparametrization([(i + 1) => Reparametrization(
    PartiallyCentered(1.0), PartiallyCentered(c), 0.0, x -> x[1] / 2)
    for (i, c) in enumerate(cs)])

const CONST_ONLY = AutoEnzyme(; function_annotation = Enzyme.Const)
const RUNTIME_ACTIVITY = AutoEnzyme(; mode = set_runtime_activity(Enzyme.Reverse),
                                      function_annotation = Enzyme.Const)
const DUPLICATED = AutoEnzyme(; function_annotation = Enzyme.Duplicated)

# Central differences on the scalar logdensity — independent of any AD system.
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

@testset "Enzyme reverse-mode" begin

    # --- site 1: the gradient hot path (`reparam_objective`) ----------------
    @testset "logdensity_and_gradient — every Enzyme variant, vs finite differences" begin
        x = 0.5 .* randn(Xoshiro(1), K + 1)
        ref = fd_gradient(ReparametrizedProblem(spec(fill(0.5, K)), Funnel(K), RUNTIME_ACTIVITY), x)
        for (name, be) in (("Const", CONST_ONLY),
                           ("Const+set_runtime_activity", RUNTIME_ACTIVITY),
                           ("Duplicated", DUPLICATED))
            rp = ReparametrizedProblem(spec(fill(0.5, K)), Funnel(K), be)
            lp, g = LogDensityProblems.logdensity_and_gradient(rp, x)
            @test lp ≈ LogDensityProblems.logdensity(rp, x)
            @test maximum(abs, g .- ref) < 1e-6
            println("  site 1  ", rpad(name, 28), "|g|=", norm(g))
        end
    end

    # --- site 2: the joint halo transport (added by `dbde8d2`) --------------
    #
    # `transport_objective` is a SECOND `value_and_gradient` call site on the
    # same user-supplied backend, and Enzyme cannot statically prove it. A plain
    # `Const` — which is correct and sufficient for site 1 — throws here, and
    # only at the first restarting warm-up window, so a user following the
    # docstring hits it partway into a run rather than at setup.
    @testset "find_reparametrization! — the joint transport site" begin
        function run_transport(be)
            rp = ReparametrizedProblem(spec(fill(1.0, K)), Funnel(K), be)
            X, G = halo(rp, 200)
            pg = DynamicHMC.evaluate_ℓ(rp, X[:, 1]; strict = false)
            find_reparametrization!(rp, X, G, pg)
            maximum(1:size(X, 2)) do j
                norm(LogDensityProblems.logdensity_and_gradient(rp, X[:, j])[2] .- G[:, j], Inf)
            end
        end

        # The regression. Promote this to a plain `@test` if `transport_objective`
        # is ever made statically provable — that is the better fix and would let
        # users pass a backend without the `mode=` incantation.
        @test_broken try
            run_transport(CONST_ONLY); true
        catch
            false
        end

        for (name, be) in (("Const+set_runtime_activity", RUNTIME_ACTIVITY),
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
