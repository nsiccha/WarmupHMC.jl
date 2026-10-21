@testitem "cooperative progress tree" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, LogDensityProblems

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "cooperative_progress.jl"))'

# `cooperative_warmup_mcmc(...; progress=node)` builds one run node under `node`
# (scheduler labels, final stop message) with one adaptive-style subtree per
# started chain (snag cooperative-prog-44129d4b, design decision 1vmqbbs).
# Before it, the sampler created no node at all.
const TB = _Treebars
p = DiagGaussian([0.5, -1.0, 0.0, 2.0], [0.5, 1.0, 1.5, 2.0])
# Skips Pathfinder. A full-matrix `squared_scale`: the diagonal form fails a
# NamedTuple init on this fixture (todo 1g03pcl).
init() = (; position=zeros(4), squared_scale=Matrix{Float64}(I, 4, 4))
chain_nodes(text) = [m.captures[1] for m in eachmatch(r"[▶✓✗] (\w+\.\d+) — ", text)]

# Throws after `limit` gradient evaluations, mid-window. `deepcopy` gives each
# chain its own counter.
mutable struct ThrowAfter
    n::Int
    limit::Int
end
LogDensityProblems.dimension(::ThrowAfter) = 4
LogDensityProblems.capabilities(::Type{ThrowAfter}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::ThrowAfter, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(t::ThrowAfter, x) = begin
    t.n += 1
    t.n > t.limit && throw(DomainError(t.n, "test target failure"))
    (-sum(abs2, x) / 2, -x)
end

@testset "run node, and one finished subtree per started chain" begin
    root = TB.initialize_progress!(:state; description="root")
    out = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:3], p;
        n_cores=2, target_ess=200, progress=root, description="coop")
    text = TB.render_text(root)
    @test occursin(r"✓ coop — [a-z_]+: joint ESS ", text)
    @test occursin("chains: — $(out.n_started) started of 3", text)
    @test occursin(r"pooled ess: — proxy \S+ of target 200 ", text)
    @test occursin(r"evaluations: — \S+ \(", text)
    @test occursin("elapsed: — ", text)
    # In index order: a chain's node is made when the scheduler reserves its index.
    @test chain_nodes(text) == ["coop.$i" for i in 1:out.n_started]
    # Each chain's six adaptive labels, as finished child lines.
    for label in ("divergent samples", "ess", "active transformation",
                  "sampling performance", "total transition counter",
                  "total evaluation counter")
        @test count(Regex("─ ✓ $label: — "), text) == out.n_started
    end
    # Nothing is left running: the run node finished and every chain node with it.
    @test all(line -> !occursin("▶", line), split(text, '\n')[2:end])
end

@testset "a progress node does not change the draws" begin
    sample(progress) = cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], p;
        n_cores=1, n_evaluations_budget=20_000, progress)
    plain = sample(nothing)
    root = TB.initialize_progress!(:state; description="root")
    watched = sample(root)
    @test [r.posterior_position for r in watched.results] ==
          [r.posterior_position for r in plain.results]
    @test watched.total_evaluation_counter == plain.total_evaluation_counter
    text = TB.render_text(root)
    @test occursin(r"✓ MCMC — eval_budget: joint ESS ", text)
    @test occursin(r"evaluations: — \S+ of 20k \(", text)
    @test occursin("pooled ess: — proxy", text) && occursin("no target_ess", text)
    @test occursin("no time budget", text)
end

@testset "a chain node shows its status, and finishes when the chain is parked" begin
    root = TB.initialize_progress!(:state; description="root")
    node = TB.initialize_progress!(root; description="MCMC.7")
    chain = WarmupHMC.cooperative_chain(Xoshiro(1), p; chain_index=7, progress=node,
        monitor_ess=true, nonlinear_adapt=false, init=init())
    @test occursin("▶ MCMC.7 — initialized, waiting for a core", TB.render_text(root))
    WarmupHMC.advance_window!(chain)
    WarmupHMC._window_progress!(chain)
    @test occursin("▶ MCMC.7 — $(chain.status) · window 1 done, waiting for a core",
                   TB.render_text(root))
    # A parked chain's labels describe the draws it keeps, as its message does
    # (after a restart they would otherwise still show the discarded epoch).
    chain.status = :stuck
    chain.stuck_reason = :geometry_stall
    chain.n_samples = 0
    chain.n_divergent_samples = 0
    WarmupHMC._window_progress!(chain)
    text = TB.render_text(root)
    @test occursin(r"✓ MCMC\.7 — parked as stuck \(geometry_stall\) after window 1: no draws retained", text)
    @test occursin("divergent samples: — 0 out of 0 ", text)
    @test occursin("ess: — no estimate from 0 draws", text)
end

@testset "resumed chains get their nodes back" begin
    d = mktempdir()
    rngs() = [Xoshiro(s) for s in 1:2]
    crashed = cooperative_warmup_mcmc(rngs(), p; n_cores=1, n_evaluations_budget=10_000,
        nonlinear_adapt=false, init=init(), checkpoint_dir=d)
    rm(joinpath(d, "run_summary.json"))   # as if the run had crashed
    root = TB.initialize_progress!(:state; description="root")
    out = cooperative_warmup_mcmc(rngs(), p; n_cores=1, n_evaluations_budget=20_000,
        nonlinear_adapt=false, init=init(), checkpoint_dir=d, resume=true, progress=root)
    text = TB.render_text(root)
    @test out.n_started >= crashed.n_started
    @test chain_nodes(text) == ["MCMC.$i" for i in 1:out.n_started]
    @test occursin(r"✓ MCMC — eval_budget: joint ESS ", text)
end

@testset "a run that throws fails its nodes" begin
    root = TB.initialize_progress!(:state; description="root")
    @test_throws CompositeException cooperative_warmup_mcmc([Xoshiro(1)], ThrowAfter(0, 500);
        n_cores=1, n_evaluations_budget=20_000, nonlinear_adapt=false, init=init(),
        progress=root)
    text = TB.render_text(root)
    @test occursin(r"✗ MCMC\b", text)
    @test occursin(r"✗ MCMC\.1 — ", text)
end
end
