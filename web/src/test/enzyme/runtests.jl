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

    # --- site 0: the documented landmine ------------------------------------
    #
    # A BARE `AutoEnzyme()` throws on any `ReparametrizedProblem` — the
    # differentiated objective closes over the reparametrizer and the frozen
    # `g_y`, which Enzyme cannot prove read-only.
    #
    # This is pinned FIRST because it is the one assertion here that can go stale
    # in the REASSURING direction. Every other test in this file fails loudly if
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
    # same user-supplied backend. Keep the plain-`Const` path here because this
    # site used to require runtime activity and failed only at the first
    # restarting warm-up window.
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

        @test run_transport(CONST_ONLY) < 1e-10

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
