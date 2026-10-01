@testitem "Treebars interrupt stops each sampler like a callback stop" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "interrupt.jl"))'

# Treebars' opt-in interrupt (`request_interrupt!` / `interrupt_requested`,
# Treebars `4c09a18`): a controller flips a progress node's flag, and every
# WarmupHMC sampler reporting into that node or below it stops at its next stop
# point — the points where a `callback` returning `true` already stops — keeping
# its partial results, with no new error path.
const TB = _Treebars
p = DiagGaussian([0.5, -1.0], [1.0, 1.5])
summary_reason(dir) = match(r"\"stop_reason\"\s*:\s*\"([a-z_]+)\"",
                            read(joinpath(dir, "run_summary.json"), String)).captures[1]

@testset "adaptive: interrupted at a window boundary, draws kept, no error" begin
    seen = Symbol[]
    r = TB.with_progress(:state; description = "probe") do probe
        adaptive_warmup_mcmc(Xoshiro(1), p; n_draws = 5000, progress = probe,
            callback = (s, stage) -> begin
                push!(seen, stage)
                stage === :window && TB.request_interrupt!(probe)
                false   # the callback never asks to stop; the interrupt does
            end)
    end
    @test seen == [:init, :window]
    @test size(r.posterior_position, 2) < 5000
end

@testset "scope: a request on one run's node leaves a sibling alone" begin
    ra, rb = TB.with_progress(:state; description = "root") do root
        TB.with_progress(root, 1; description = "a") do a
            TB.with_progress(root, 1; description = "b") do b
                TB.request_interrupt!(a)
                (adaptive_warmup_mcmc(Xoshiro(2), p; n_draws = 200, progress = a),
                 adaptive_warmup_mcmc(Xoshiro(3), p; n_draws = 200, progress = b))
            end
        end
    end
    @test size(ra.posterior_position, 2) < 200    # stopped at its :init boundary
    @test size(rb.posterior_position, 2) >= 200   # ran to completion
end

@testset "progress = nothing is never interrupted" begin
    r = adaptive_warmup_mcmc(Xoshiro(4), p; n_draws = 100, progress = nothing)
    @test size(r.posterior_position, 2) >= 100
end

@testset "completion: no slot starts after an interrupt; same quorum error as a callback stop" begin
    err = try
        TB.with_progress(:state; description = "probe") do probe
            TB.request_interrupt!(probe)
            completion_warmup_mcmc([Xoshiro(31), Xoshiro(32)], p;
                n_draws = 40, min_completed = 2, progress = probe)
        end
        nothing
    catch e
        e
    end
    @test err isa WarmupHMC.CompletionQuorumError
    @test err.outcome.completion.n_started == 0
end

@testset "cooperative: stop_reason = interrupted" begin
    d = mktempdir()
    TB.with_progress(:state; description = "probe") do probe
        TB.request_interrupt!(probe)
        cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], p;
            n_cores = 1, n_evaluations_budget = 10_000, nonlinear_adapt = false,
            checkpoint_dir = d, progress = probe)
    end
    @test summary_reason(d) == "interrupted"
end

@testset "clustered: stop_reason = interrupted, no window stepped" begin
    d = mktempdir()
    out = TB.with_progress(:state; description = "probe") do probe
        TB.request_interrupt!(probe)
        clustered_warmup_mcmc(Xoshiro.(1:3), p; n_draws = 100, max_windows = 5,
            parallel = false, checkpoint_dir = d, progress = probe)
    end
    @test summary_reason(d) == "interrupted"
end
end
