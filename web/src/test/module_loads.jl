@testitem "the module loads" begin
    using WarmupHMC
    @test isdefined(WarmupHMC, :WarmupHMC)

    # The exported surface, pinned by name. `using WarmupHMC` succeeding says only
    # that the package precompiled; it says nothing about whether the eight names
    # consumers actually call are still exported. A rename that leaves the method
    # reachable as `WarmupHMC.foo` breaks every caller and nothing else here would
    # notice, because the rest of the suite reaches internals through explicit
    # `WarmupHMC:` imports.
    for name in (:adaptive_warmup_mcmc, :resume_warmup_mcmc, :cooperative_warmup_mcmc,
                 :clustered_warmup_mcmc, :ReparametrizedProblem,
                 :IndexedReparametrization, :PartiallyCentered, :Reparametrization)
        @test name in names(WarmupHMC)
    end
end
