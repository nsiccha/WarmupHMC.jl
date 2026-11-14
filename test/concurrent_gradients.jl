@testitem "concurrent chains refuse reparametrization gradients that are unsafe beside other threads" begin
    using Test, WarmupHMC
    using DifferentiationInterface, Enzyme
    const WH = WarmupHMC

    # Registered Enzyme compiles gradients that a garbage collection started by
    # another thread can break; dogfood from 72d5d25 carries the repair (snag
    # `warmuphmc-concur-06de2cf4`). The extension asks the loaded Enzyme.
    repaired = isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass)
    for backend in (AutoEnzyme(), AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Forward))
        reason = WH._unsafe_concurrent_gradients(backend)
        @test isnothing(reason) == repaired
        repaired || @test occursin("RestoreSafepointEffectsPass", reason)
    end
    @test isnothing(WH._unsafe_concurrent_gradients(nothing))

    # The refusal fires only for chains that run at once, which needs two
    # threads; the suite usually runs on one.
    cases = joinpath(@__DIR__, "concurrent_gradient_cases.jl")
    program = "include(" * repr(cases) * "); using .ConcurrentGradientCases; " *
        "ConcurrentGradientCases.run_cases()"
    @test success(run(`$(Base.julia_cmd()) --threads=2 --startup-file=no --project=$(dirname(Base.active_project())) -e $program`))
end
