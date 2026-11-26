# Snag collapsed-chains-10d2df94: on a 20-chain completion fit, 4 chains ran for
# hours at step sizes of 1e-11 to 1e-7 with metric scales near 1e-9, every
# transition at the tree-depth ceiling, `NaN` ESS and `Inf` "marginal scale
# changes". The step-size check judged a chain only once it reached `n_draws`
# (11 hours away), and a quorum stop waits for the end of the current window.
# A chain still warming up is now judged at every window boundary, its own and
# its siblings', by its step size at its last two window boundaries against the
# siblings that keep draws, and is stopped at its next transition once left out.
@testitem "completion leaves out a chain that collapses in warm-up" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems

    fixture(stepsize; n=20) = (; posterior_position=zeros(2, n), n_divergent_samples=0, stepsize)
    checks = WarmupHMC._CompletionAdmission(10.0, 0.1, Inf)
    # The warm-up threshold: `stuck_stepsize_ratio` times the margin.
    factor = 10.0 * WarmupHMC._WARMUP_STEPSIZE_MARGIN

    @testset "judged at its own window boundary" begin
        s = WarmupHMC._CompletionState(3; clock=() -> 0.0, min_completed=2, admission=checks)
        WarmupHMC._completion_worker!(s, 1) do _, stop1
            WarmupHMC._report_boundary!(stop1, 0.2, 30; window=true)
            WarmupHMC._completion_worker!(s, 2) do _, stop2
                # Initialization is not a window: its step size is not judged.
                WarmupHMC._report_boundary!(stop2, 1e-9, 0)
                @test s.statuses[2] === :running
                @test !stop2()
                # One window boundary is not enough: a chain can dip and recover.
                WarmupHMC._report_boundary!(stop2, 1e-9, 0; window=true)
                @test s.statuses[2] === :running
                # Two are, with or without draws.
                WarmupHMC._report_boundary!(stop2, 1e-9, 0; window=true)
                @test s.statuses[2] === :stuck
                @test s.admissions[2].stuck_reason === :stepsize
                @test s.admissions[2].stepsize == 1e-9
                @test s.admissions[2].reference_stepsize == 0.2
                @test isnothing(s.admissions[2].divergent_fraction)
                # Told to stop, at the boundary and between transitions.
                @test stop2()
                @test WarmupHMC._left_out(stop2)
                @test !stop1()
                fixture(1e-9; n=3), false
            end
            # The verdict survives the chain's return.
            @test s.statuses[2] === :stuck
            fixture(0.2), true
        end
        WarmupHMC._completion_worker!((_, _) -> (fixture(0.21), true), s, 3)
        @test s.statuses == [:completed, :stuck, :completed]
    end

    @testset "judged at a sibling's boundary, then stopped between transitions" begin
        s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=1, admission=checks)
        WarmupHMC._completion_worker!(s, 2) do _, stop2
            WarmupHMC._completion_worker!(s, 1) do _, stop1
                # No sibling keeps draws yet: nothing to compare with.
                WarmupHMC._report_boundary!(stop2, 2e-9, 0; window=true)
                WarmupHMC._report_boundary!(stop2, 1e-9, 0; window=true)
                @test s.statuses[2] === :running
                # A sibling's first boundary with draws settles it.
                WarmupHMC._report_boundary!(stop1, 0.2, 30; window=true)
                @test s.statuses[2] === :stuck
                fixture(0.2), true
            end
            @test WarmupHMC._left_out(stop2)
            fixture(1e-9; n=0), false
        end
        @test s.statuses == [:completed, :stuck]
    end

    @testset "within the margin it keeps warming up" begin
        s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=2, admission=checks)
        WarmupHMC._completion_worker!(s, 1) do _, stop1
            WarmupHMC._report_boundary!(stop1, 0.2, 30; window=true)
            WarmupHMC._completion_worker!(s, 2) do _, stop2
                WarmupHMC._report_boundary!(stop2, 1e-9, 0; window=true)
                WarmupHMC._report_boundary!(stop2, 0.2 / factor * 1.01, 0; window=true)
                @test s.statuses[2] === :running
                # A chain that dipped and recovered keeps warming up.
                WarmupHMC._report_boundary!(stop2, 1e-9, 0; window=true)
                WarmupHMC._report_boundary!(stop2, 0.15, 0; window=true)
                @test s.statuses[2] === :running
                @test !stop2()
                # Finished, the ordinary check at `n_draws` still applies.
                fixture(0.004), true
            end
            @test s.statuses[2] === :stuck
            @test !isnothing(s.admissions[2].divergent_fraction)
            fixture(0.2), true
        end
        @test s.statuses == [:completed, :stuck]
    end

    @testset "switched off with the step-size check" begin
        off = WarmupHMC._CompletionAdmission(Inf, 0.1, Inf)
        s = WarmupHMC._CompletionState(2; clock=() -> 0.0, min_completed=2, admission=off)
        WarmupHMC._completion_worker!(s, 1) do _, stop1
            WarmupHMC._report_boundary!(stop1, 0.2, 30; window=true)
            WarmupHMC._completion_worker!(s, 2) do _, stop2
                WarmupHMC._report_boundary!(stop2, 1e-12, 0; window=true)
                WarmupHMC._report_boundary!(stop2, 1e-12, 0; window=true)
                @test !stop2()
                fixture(1e-12), true
            end
            fixture(0.2), true
        end
        @test s.statuses == [:completed, :completed]
    end

    @testset "a collapsed chain that keeps draws is no reference" begin
        s = WarmupHMC._CompletionState(4; clock=() -> 0.0, min_completed=1, admission=checks)
        WarmupHMC._completion_worker!(s, 1) do _, stop1
            WarmupHMC._completion_worker!(s, 2) do _, stop2
                WarmupHMC._completion_worker!(s, 3) do _, stop3
                    WarmupHMC._report_boundary!(stop1, 0.2, 30; window=true)
                    WarmupHMC._report_boundary!(stop2, 0.22, 30; window=true)
                    # Collapsed, with draws: left out, and dropped as a reference.
                    WarmupHMC._report_boundary!(stop3, 1e-9, 30; window=true)
                    WarmupHMC._report_boundary!(stop3, 1e-9, 30; window=true)
                    @test s.statuses[3] === :stuck
                    @test isnothing(s.sampling_stepsizes[3])
                    fixture(1e-9; n=30), false
                end
                fixture(0.22), true
            end
            fixture(0.2), true
        end
        @test s.admissions[3].reference_stepsize ≈ 0.21
    end

    @testset "named in the outcome and in the quorum error" begin
        entered = Channel{Int}(3)
        release = Channel{Nothing}(3)
        batch = Threads.@spawn WarmupHMC._completion_batch(3; min_completed=3,
                admission=checks) do i, stop
            if i == 3
                put!(entered, i)
                take!(release)
                WarmupHMC._report_boundary!(stop, 1e-9, 0; window=true)
                WarmupHMC._report_boundary!(stop, 1e-10, 0; window=true)
                @test stop()
                return fixture(1e-10; n=0), false
            end
            WarmupHMC._report_boundary!(stop, 0.2, 30; window=true)
            put!(entered, i)
            take!(release)
            fixture(0.2), true
        end
        foreach(_ -> take!(entered), 1:3)
        foreach(_ -> put!(release, nothing), 1:3)
        out = fetch(batch)
        c = out.completion
        @test !c.quorum_met
        @test c.stuck_chain_indices == [3]
        @test c.n_stuck == 1
        @test c.chains[3].status === :stuck
        @test c.chains[3].stuck_reason === :stepsize
        @test c.chains[3].stepsize == 1e-10
        @test c.chains[3].reference_stepsize == 0.2
        @test isnothing(c.chains[3].divergent_fraction)
        @test c.chains[3].n_retained_samples == 0
        @test c.selection_warning == ""
        message = sprint(showerror, WarmupHMC.CompletionQuorumError(out))
        @test occursin("left out during warm-up, their step size collapsed below " *
                       "their siblings': [3]", message)
        @test !occursin("after reaching `n_draws`", message)
    end

    @testset "resume: a collapsed partial checkpoint does not restart" begin
        out = WarmupHMC._completion_batch(3; min_completed=1, admission=checks,
                initial_stepsizes=Dict(1 => 0.2, 2 => 0.21),
                initial_warming_stepsizes=Dict(1 => (0.2, 0.2), 2 => (0.21, 0.21),
                    3 => (1e-9, 1e-10))) do i, stop
            i == 3 && error("slot 3 must not restart")
            fixture(0.2), true
        end
        @test out.completion.chains[3].status === :stuck
        @test out.completion.chains[3].reference_stepsize ≈ 0.205
    end

    # End to end: a chain whose log density is noisy at Float64 resolution (no
    # step size reaches the target acceptance there, as with a cancelling formula
    # or an adaptive solver's tolerance noise) collapses, and is left out during
    # warm-up instead of running towards `n_draws`. Its density is noisy
    # everywhere, so it cannot escape whatever the seed; its siblings' nowhere.
    @testset "end to end" begin
        healthy = NoisyTrapProblem(6, Inf, 5.0)
        trapped = NoisyTrapProblem(6, -Inf, 5.0)
        start = (; position=zeros(6), squared_scale=ones(6))
        err = try
            completion_warmup_mcmc([Xoshiro(11), Xoshiro(12), Xoshiro(13)],
                [healthy, healthy, trapped]; n_draws=200, min_completed=3, init=start)
        catch e
            e
        end
        @test err isa WarmupHMC.CompletionQuorumError
        c = err.outcome.completion
        @test c.chains[3].status === :stuck
        @test c.chains[3].stuck_reason === :stepsize
        @test c.chains[3].stepsize < c.chains[3].reference_stepsize / factor
        @test c.completed_chain_indices == [1, 2]
        # Stopped long before 200 draws.
        @test c.chains[3].n_samples < 200
        @test c.chains[3].n_warmup_windows >= 2
        @test c.chains[3].gradient_calls >= c.chains[3].total_evaluation_counter > 0
        @test occursin("left out during warm-up", sprint(showerror, err))
    end

    # A window whose recorded positions never moved in a coordinate has no finite
    # restart factor. The progress label reads `frozen`, never `Inf`, and the run
    # warns once (primer 1349378f's reproduction: a squared scale of 2^-140
    # freezes coordinate 2 at its first window).
    @testset "a frozen coordinate is labelled and warned about once" begin
        p = NoisyTrapProblem(2, Inf, 0.0)
        labels = String[]
        r = @test_logs (:warn, r"did not move in 1 of 2 coordinates") match_mode=:any begin
            adaptive_warmup_mcmc(Xoshiro(3), p; n_draws=100,
                init=(; position=[0.3, 0.2], squared_scale=[1.0, 2.0^-140]),
                callback=(st, stage) -> (push!(labels, string(WarmupHMC.ActiveTransformation(
                    st.kinetic_energy, st.scale_changes))); false))
        end
        @test any(!isfinite, r.scale_changes)
        @test any(l -> occursin("frozen", l), labels)
        @test !any(l -> occursin("Inf", l), labels)
        @test _Treebars.short_string(WarmupHMC._ScaleChange.([Inf, 2.0, NaN])) ==
              "[frozen, 2.0, frozen]"
        # A run whose record already holds a frozen window does not warn again.
        state = (; variance_cond=Inf, scale_changes=[Inf], variance_memory=[0.0, 1.0],
            outer_counter=2, dimension=2, stepsize=0.1)
        @test_logs WarmupHMC._warn_frozen(state)
        @test_logs (:warn, r"did not move in 1 of 2") WarmupHMC._warn_frozen(
            merge(state, (; scale_changes=Float64[])))
        # A finite restart factor never warns.
        @test_logs WarmupHMC._warn_frozen(merge(state, (; variance_cond=4.0, scale_changes=Float64[])))
    end
end
