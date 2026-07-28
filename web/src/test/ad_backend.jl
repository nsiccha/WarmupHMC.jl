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

# Loading DifferentiationInterface must actually have triggered the extension;
# without it every `ReparametrizedProblem` gradient throws, and a test that only
# ever exercised `logdensity` would not notice.
isnothing(Base.get_extension(WarmupHMC, :DifferentiationInterfaceExt)) && error(
    "DifferentiationInterface is loaded but WarmupHMC's DifferentiationInterfaceExt " *
    "did not activate — ReparametrizedProblem gradients would error."
)

const TEST_AD_LOADED = true
end
