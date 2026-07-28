# The behavioural gate that used to abort `ad_backend.jl` at load time.
#
# Everything else in that file was scaffolding for a problem this suite no longer
# has: with no test `Project.toml`, the suite ran under `--project=.` where
# ForwardDiff is not a direct dependency, so it had to be pulled in with
# `Base.require(Base.PkgId(UUID(...), "ForwardDiff"))`. There is a real test
# environment now (`web/src/test/Project.toml`) and `using ForwardDiff` is all it
# takes — see the `ADBackend` snippet in `setup.jl`.
#
# The gate itself is worth keeping, so here it is as a test rather than as a
# side effect of loading a file.

@testitem "the reparametrized gradient path computes a gradient" setup=[ADBackend, Determinism] tags=[:reparametrization] begin
    using WarmupHMC, LogDensityProblems

    # A trivial target: coordinate 2 is reparametrized and reads its log-scale off
    # coordinate 1, so this exercises the real closure path.
    struct ADGateTarget end
    LogDensityProblems.dimension(::ADGateTarget) = 2
    LogDensityProblems.capabilities(::Type{ADGateTarget}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(::ADGateTarget, x) = -sum(abs2, x) / 2
    LogDensityProblems.logdensity_and_gradient(::ADGateTarget, x) = (-sum(abs2, x) / 2, -x)

    # ASSERT THE BEHAVIOUR, NOT THE MECHANISM. Two structural checks have stood
    # here and both were unfalsifiable, in opposite directions — worth recording,
    # because the third one will look reasonable too:
    #
    #   `isnothing(Base.get_extension(WarmupHMC, :DifferentiationInterfaceExt))`
    #       pinned the suite to WHERE the method lives. When the method moved from
    #       `ext/` into `src/` — no observable change — it aborted the WHOLE suite
    #       with a message blaming the AD backend. It failed on a non-event.
    #
    #   `hasmethod(WarmupHMC._logdensity_and_gradient_reparam, ...)`
    #       has the opposite defect: it can never fail. That method is defined
    #       unconditionally in `src/Reparametrizations.jl`, with
    #       DifferentiationInterface a hard `[deps]` entry — no weakdep, no
    #       conditional compilation. Asking whether it exists is asking whether
    #       the file we just loaded loaded.
    #
    # So: call it. That is strictly stronger than either — it catches a method
    # that resolves but computes the wrong thing, and it is indifferent to which
    # module the method ends up in, which is precisely the axis that has churned.
    rp = WarmupHMC.ReparametrizedProblem(
        WarmupHMC.IndexedReparametrization([
            2 => WarmupHMC.Reparametrization(
                WarmupHMC.PartiallyCentered(1.0), WarmupHMC.PartiallyCentered(0.5),
                0.0, x -> x[1] / 2),
        ]),
        ADGateTarget(), AutoForwardDiff())

    lp, g = LogDensityProblems.logdensity_and_gradient(rp, [0.3, -0.2])
    @test isfinite(lp)
    @test length(g) == 2
    @test all(isfinite, g)

    # Non-vacuity: a gradient of all zeros would satisfy everything above while
    # meaning the closure never differentiated anything.
    @test any(!=(0), g)
end
