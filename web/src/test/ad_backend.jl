# Shared loader for the AD stack the reparametrization tests need.
#
# `ReparametrizedProblem` computes gradients through DifferentiationInterface,
# which IS a direct dependency of WarmupHMC. The *backend* is not: these tests
# pin `AutoForwardDiff`, which needs ForwardDiff in-session, and ForwardDiff
# arrives only transitively (via Pathfinder), so a plain `using ForwardDiff`
# fails under `--project=.`, which is how this suite is actually run:
#
#     ERROR: ArgumentError: Package ForwardDiff not found in current path.
#
# That is why `golden_awm.jl` could not be included from `runtests.jl` as
# written. Loading by UUID goes through the same machinery as `using` but
# resolves against the *manifest* rather than the project's direct dependencies,
# which works under both `--project=.` and `Pkg.test()`.
#
# ForwardDiff is the backend the FROZEN GOLDEN BASELINES were recorded against,
# which is the only reason it is pinned here — it is not a recommendation. New
# code should use a reverse-mode backend (`AutoEnzyme()`); see the
# `ReparametrizedProblem` docstring.
#
# It is deliberately NOT tolerant: a missing backend must abort the suite, not
# quietly skip the tests that need it. A reparametrization suite that silently
# stops testing reparametrization is worse than no suite at all.
if !@isdefined(TEST_AD_LOADED)

using UUIDs: UUID

_require_pkg(name, uuid) = try
    Base.require(Base.PkgId(UUID(uuid), name))
catch err
    error("""
    The reparametrization tests need `$name` loadable in this environment, but
    it could not be required:

        $(sprint(showerror, err))

    `$name` should be reachable through the resolved manifest. Re-resolve this
    worktree (see the repo's canonical resolve) and try again.
    """)
end

const ForwardDiff = _require_pkg("ForwardDiff", "f6369f11-7733-5829-9624-2563aa707210")
const DifferentiationInterface = _require_pkg("DifferentiationInterface", "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63")
const AutoForwardDiff = DifferentiationInterface.AutoForwardDiff

# The gradient method must exist. It used to live in a package extension, so this
# guarded against the extension failing to activate; now that
# DifferentiationInterface is a direct dependency, that failure mode is a
# precompile error instead — but the method can still go missing by accident, and
# a suite that only ever exercised `logdensity` would not notice.
hasmethod(WarmupHMC._logdensity_and_gradient_reparam,
          Tuple{WarmupHMC.ReparametrizedProblem, Vector{Float64}}) || error(
    "WarmupHMC._logdensity_and_gradient_reparam has no method for " *
    "(ReparametrizedProblem, Vector{Float64}) — ReparametrizedProblem gradients would error."
)

const TEST_AD_LOADED = true
end
