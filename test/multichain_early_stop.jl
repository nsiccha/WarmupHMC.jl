@testitem "multi-chain run stopped early returns per-chain results" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random
    const MCMCDiagnosticTools = WarmupHMC.MCMCDiagnosticTools

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "multichain_early_stop.jl"))'

# A `callback` stop or a Treebars interrupt ends each chain at its next boundary,
# so the chains of one multi-chain run retain DIFFERENT draw counts — zero when
# stopped at `:init`. With a progress node the multi-chain method used to pool
# them with `stack`, which threw `DimensionMismatch` on any such run (snag
# `multichain-zero-b00caec9`): a user's Stop surfaced as a failed fit.
const TB = _Treebars
p = DiagGaussian(collect(range(-1.0, 1.0, length = 5)), collect(range(0.5, 2.0, length = 5)))
draw_counts(rv) = [size(r.posterior_position, 2) for r in rv]

# Runs `f(probe)` under a `:state` root; returns its value and the rendered tree.
function with_tree(f)
    tree = Ref("")
    value = TB.with_progress(:state; description = "probe") do probe
        v = f(probe)
        tree[] = TB.render_text(probe)
        v
    end
    value, tree[]
end

@testset "stopped at :init: zero draws per chain, parent finalized" begin
    rv, tree = with_tree() do probe
        adaptive_warmup_mcmc(Xoshiro.(1:3), p; n_draws = 300, progress = probe,
            callback = (s, stage) -> true)
    end
    @test length(rv) == 3
    @test draw_counts(rv) == [0, 0, 0]
    @test occursin("✓ MCMC (3/3) — no draws retained", tree)
    @test !occursin("✗", tree)
end

@testset "a Treebars interrupt before sampling behaves the same" begin
    rv, tree = with_tree() do probe
        TB.request_interrupt!(probe)
        adaptive_warmup_mcmc(Xoshiro.(1:3), p; n_draws = 300, progress = probe)
    end
    @test draw_counts(rv) == [0, 0, 0]
    @test occursin("no draws retained", tree)
end

@testset "ragged stop: ESS pools the common tail and says so" begin
    # Sequential chains; chain c stops at its (c-1)-th boundary (`:init` is the 0th),
    # so chain 1 keeps no draws and the others keep different counts.
    chain, boundary = Ref(0), Ref(0)
    stop = (s, stage) -> begin
        stage === :init && (chain[] += 1; boundary[] = 0)
        done = boundary[] >= chain[] - 1
        boundary[] += 1
        done
    end
    rv, tree = with_tree() do probe
        adaptive_warmup_mcmc(Xoshiro.(1:3), p; n_draws = 5000, parallel = false,
            progress = probe, callback = stop)
    end
    counts = draw_counts(rv)
    @test counts[1] == 0
    @test all(n -> 10 < n < 5000, counts[2:3])
    @test counts[2] != counts[3]
    m = minimum(counts[2:3])
    stacked = WarmupHMC._stack_chain_tails(getproperty.(rv, :posterior_position); min_chain_draws = 11)
    @test size(stacked) == (m, 2, 5)
    for (j, r) in enumerate(rv[2:3])
        @test stacked[:, j, :] == r.posterior_position[:, end-m+1:end]'
    end
    message = WarmupHMC._multichain_message(rv, true)
    @test startswith(message, "min. ESS: $(WarmupHMC.short_string(minimum(MCMCDiagnosticTools.ess(stacked)))) (last $m draws of 2/3 chains), divergent: ")
    @test occursin("✓ MCMC (3/3) — $message", tree)
end

@testset "equal-length chains: message unchanged from the stack-based form" begin
    rv, tree = with_tree() do probe
        adaptive_warmup_mcmc(Xoshiro.(1:3), p; n_draws = 100, progress = probe)
    end
    @test allequal(draw_counts(rv))
    old = "min. ESS: $(WarmupHMC.short_string(minimum(MCMCDiagnosticTools.ess(permutedims(stack(getproperty.(rv, :posterior_position)), (2, 3, 1)))))), " *
          "divergent: $(WarmupHMC.short_string(100 * sum(r -> r.n_divergent_samples, rv) / sum(draw_counts(rv))))%"
    @test WarmupHMC._multichain_message(rv, true) == old
    @test occursin("✓ MCMC (3/3) — $old", tree)
end

@testset "_stack_chain_tails: exclusion, tail truncation, no aliasing" begin
    a, b, c = reshape(1.0:12.0, 2, 6), reshape(101.0:108.0, 2, 4), zeros(2, 2)
    s = WarmupHMC._stack_chain_tails([a, b, c]; min_chain_draws = 3)
    @test size(s) == (4, 2, 2)                    # c excluded; a truncated to its last 4
    @test s[:, 1, :] == a[:, 3:6]'
    @test s[:, 2, :] == b'
    s[1, 1, 1] = -1.0
    @test a[1, 3] == 5.0
    @test isnothing(WarmupHMC._stack_chain_tails([c, c]; min_chain_draws = 3))
    @test isnothing(WarmupHMC._stack_chain_tails([zeros(2, 3), a]; min_chain_draws = 1))   # m = 3
    @test isnothing(WarmupHMC._stack_chain_tails(Matrix{Float64}[]; min_chain_draws = 1))
end
end
