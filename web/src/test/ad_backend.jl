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
using LogDensityProblems

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

# A trivial target for the gate below: coordinate 2 is reparametrized and reads
# its log-scale off coordinate 1, so the gate exercises the real closure path.
struct _ADGateTarget end
LogDensityProblems.dimension(::_ADGateTarget) = 2
LogDensityProblems.capabilities(::Type{_ADGateTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_ADGateTarget, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::_ADGateTarget, x) = (-sum(abs2, x) / 2, -x)

# The gradient path must actually WORK. Assert the BEHAVIOUR, not the mechanism.
#
# Two structural checks have stood here and both were unfalsifiable, in opposite
# directions — worth recording, because the third one will look reasonable too:
#
#   `isnothing(Base.get_extension(WarmupHMC, :DifferentiationInterfaceExt))`
#       pinned the suite to WHERE the method lives. When the method moved from
#       `ext/` into `src/` — no observable change — this line aborted the WHOLE
#       suite (`runtests.jl` includes this file before any testset) with a
#       message blaming the AD backend. It failed on a non-event.
#
#   `hasmethod(WarmupHMC._logdensity_and_gradient_reparam, ...)`
#       has the opposite defect: it can never fail. That method is now defined
#       unconditionally in `src/Reparametrizations.jl`, with
#       DifferentiationInterface a hard `[deps]` entry — no weakdep, no
#       conditional compilation. Asking whether it exists is asking whether the
#       file we just loaded loaded. It cannot distinguish a working gradient
#       path from a broken one, which is the only thing worth knowing here.
#
# So: call it. That is strictly stronger than either — it catches a method that
# resolves but computes the wrong thing, and it is indifferent to which module
# the method ends up in, which is precisely the axis that has already churned.
#
# IF YOU COPY THIS AS A TEMPLATE, DO NOT SWAP IN A BARE `AutoEnzyme()`. It throws
# on any `ReparametrizedProblem` — the differentiated objective is a closure over
# the reparametrizer and the frozen `g_y`, which Enzyme cannot prove read-only:
#
#     EnzymeMutabilityException: Function argument passed to autodiff cannot be
#     proven readonly
#
# The fix is `AutoEnzyme(; function_annotation = Enzyme.Const)`. Enzyme's own
# error text suggests `Duplicated` instead; that agrees to machine precision and
# is ~22x slower, so the hint diagnoses the problem without being the fix. See
# the `ReparametrizedProblem` docstring in `src/Reparametrizations.jl`.
#
# `AutoForwardDiff` here is not a recommendation either — it is what the frozen
# golden baselines were recorded against (see the header). NOTE that this means
# the documented Enzyme guidance above is asserted by NO test: Enzyme is not in
# this manifest, so nothing in this suite can execute it.
let rp = WarmupHMC.ReparametrizedProblem(
        WarmupHMC.IndexedReparametrization([
            2 => WarmupHMC.Reparametrization(
                WarmupHMC.PartiallyCentered(1.0), WarmupHMC.PartiallyCentered(0.5),
                0.0, x -> x[1] / 2),
        ]),
        _ADGateTarget(), AutoForwardDiff())
    g = try
        LogDensityProblems.logdensity_and_gradient(rp, [0.3, -0.2])[2]
    catch err
        error("""
        `ReparametrizedProblem` cannot compute a gradient in this environment:

            $(sprint(showerror, err))

        ForwardDiff and DifferentiationInterface both loaded above, so the AD
        stack itself is fine — it is the reparametrized gradient path that is
        broken. Every reparametrization testset below would fail; aborting here
        so the cause is legible rather than arriving as a wall of red.
        """)
    end
    length(g) == 2 && all(isfinite, g) || error(
        "ReparametrizedProblem gradient returned $(g) — expected 2 finite values. " *
        "The gradient path resolves but computes the wrong thing."
    )
end

const TEST_AD_LOADED = true
end
