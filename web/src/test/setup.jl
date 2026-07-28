# Shared `@testsnippet`s. A snippet's code is spliced into each requesting item's
# own module, so the old `if !@isdefined(...)` include-guards no longer do any
# work: nothing is ever defined twice, and one item cannot see another's
# definitions. Request them with `setup=[Targets]` etc. on the `@testitem`.
#
# `@testsnippet` rather than `@testmodule` deliberately: a module would put every
# name behind a qualifier, so `Funnel(3)` at its ~40 existing call sites would
# have to become `Targets.Funnel(3)`. A snippet keeps the names unqualified,
# which is what makes converting this suite a mechanical wrap.

# --- Determinism -------------------------------------------------------------
@testsnippet Determinism begin
    using LinearAlgebra: BLAS
    # The adaptive transformation update runs multithreaded BLAS, whose reduction
    # order is not deterministic run-to-run. Every byte-identity comparison in
    # this suite is verified under this pin.
    #
    # `runtests.jl` sets this too, for the whole-suite run. This copy is what
    # makes an item honest when it is run ALONE — from the VS Code test explorer,
    # or via `--htmxo-test=` — where `runtests.jl` never executes. An item that
    # compares draws for byte-identity must list `Determinism` in its `setup`.
    BLAS.set_num_threads(1)
end

# --- AD backend --------------------------------------------------------------
@testsnippet ADBackend begin
    using ForwardDiff, DifferentiationInterface

    # `ReparametrizedProblem` computes gradients through DifferentiationInterface,
    # which IS a direct dependency of WarmupHMC. The *backend* is not: WarmupHMC
    # is backend-agnostic and the caller supplies one.
    #
    # `AutoForwardDiff` is pinned across this suite because it is what the FROZEN
    # GOLDEN BASELINES in `golden_awm.jl` were recorded against. That is the only
    # reason — it is NOT a recommendation. New code should use a reverse-mode
    # backend; see the `ReparametrizedProblem` docstring, and `enzyme.jl` for what
    # that actually requires in practice.
    #
    # This snippet replaces `ad_backend.jl`, whose entire `Base.require`-by-UUID
    # loader existed only because there was no test `Project.toml`: ForwardDiff is
    # not a direct dependency of WarmupHMC (it arrives transitively via
    # Pathfinder), so a plain `using ForwardDiff` failed under `--project=.`.
    # There is a test environment now and a plain `using` is all it takes. That
    # file's behavioural gate survives as its own item in `ad_backend_gate.jl`.
end

# --- Targets -----------------------------------------------------------------
@testsnippet Targets begin
    # Self-contained test targets — no BridgeStan, no PosteriorDB, all with
    # analytic gradients so the target itself never needs an AD backend (the
    # reparametrization wrapper supplies its own).
    #
    # `include`d rather than inlined here so `targets.jl` stays usable from a
    # plain script — `golden_awm_capture.jl` needs the same targets and runs
    # outside any test runner.
    include(joinpath(@__DIR__, "targets.jl"))
end
