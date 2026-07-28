# Shared loader for the AD stack the reparametrization tests need.
#
# `ReparametrizedProblem` computes gradients through `WarmupHMC`'s
# `DifferentiationInterfaceExt` extension, which only exists once
# DifferentiationInterface is loaded; the backend itself (`AutoForwardDiff`)
# additionally needs ForwardDiff in-session. Neither is a dependency of
# `WarmupHMC` proper — DifferentiationInterface is a `[weakdeps]` entry and
# ForwardDiff arrives transitively via Pathfinder — so a plain
# `using ForwardDiff` fails under `--project=.`, which is how this suite is
# actually run:
#
#     ERROR: ArgumentError: Package ForwardDiff not found in current path.
#
# That is why `golden_awm.jl` could not be included from `runtests.jl` as
# written. Loading by UUID goes through the same machinery as `using` (so the
# extension is triggered normally) but resolves against the *manifest* rather
# than the project's direct dependencies, which works under both `--project=.`
# and `Pkg.test()`.
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
    worktree (see the repo's canonical resolve) and try again. Do NOT `Pkg.add`
    DifferentiationInterface into this project — it is a `[weakdeps]` entry and
    promoting it deletes the `[weakdeps]` section, orphaning `[extensions]`.
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
# This gate used to read
# `isnothing(Base.get_extension(WarmupHMC, :DifferentiationInterfaceExt)) && error(...)`,
# which pins the suite to WHERE the method is defined rather than to whether it
# exists. Moving `_logdensity_and_gradient_reparam` out of `ext/` and into `src/`
# changes nothing observable, but would abort the WHOLE suite on this line —
# `runtests.jl` includes this file before any testset — with a message blaming
# the AD backend.
#
# `hasmethod` is no better: `src/Reparametrizations.jl` defines a fallback
# `_logdensity_and_gradient_reparam` that throws "requires DifferentiationInterface",
# so a method always exists and the check can never fail. Both structural checks
# are wrong for the same reason. Calling it is the only thing that distinguishes
# a working gradient path from a missing one — and it is strictly stronger, since
# it also catches an extension that loads but returns the wrong shape.
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
        stack itself is fine — the reparametrized gradient METHOD is unreachable.
        Either WarmupHMC's `DifferentiationInterfaceExt` did not activate, or the
        method it provides moved and nothing replaced it. Every reparametrization
        testset below would fail; aborting here so the cause is legible.
        """)
    end
    length(g) == 2 && all(isfinite, g) || error(
        "ReparametrizedProblem gradient returned $(g) — expected 2 finite values. " *
        "The gradient path resolves but computes the wrong thing."
    )
end

const TEST_AD_LOADED = true
end
