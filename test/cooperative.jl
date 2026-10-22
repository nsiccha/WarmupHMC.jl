@testitem "cooperative_warmup_mcmc" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems, Serialization
    using WarmupHMC: CoopMoments, _fit_moments!, _combine_moments!, _moment_scales

# Focused invocation from the package root:
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "cooperative.jl"))'

# The merged cooperative sampler (decisions 0m327rr / 0m8p12j): pooled,
# accumulating moments through soft clusters, the original's √2 restart rule,
# rounds and doubling batches, one seed RNG.

# Two Gaussian modes 100 units apart in every coordinate, with different scales:
# chains started in one never reach the other, so each mode's chains should pool
# only with each other.
struct TwoModes
    sigma_a::Vector{Float64}
    sigma_b::Vector{Float64}
end
const MODE = 50.0
LogDensityProblems.dimension(p::TwoModes) = length(p.sigma_a)
LogDensityProblems.capabilities(::Type{TwoModes}) = LogDensityProblems.LogDensityOrder{1}()
_mode_terms(p::TwoModes, x) = begin
    za = (x .+ MODE) ./ p.sigma_a
    zb = (x .- MODE) ./ p.sigma_b
    la = -sum(abs2, za) / 2 - sum(log, p.sigma_a)
    lb = -sum(abs2, zb) / 2 - sum(log, p.sigma_b)
    (; za, zb, la, lb)
end
LogDensityProblems.logdensity(p::TwoModes, x) = ((; la, lb) = _mode_terms(p, x); max(la, lb) + log1p(exp(-abs(la - lb))))
LogDensityProblems.logdensity_and_gradient(p::TwoModes, x) = begin
    (; za, zb, la, lb) = _mode_terms(p, x)
    l = max(la, lb) + log1p(exp(-abs(la - lb)))
    wa = exp(la - l)
    (l, -wa .* za ./ p.sigma_a .- (1 - wa) .* zb ./ p.sigma_b)
end

target = DiagGaussian(collect(range(-1.0, 1.0, 10)), 10 .^ collect(range(-2.0, 2.0, 10)))

@testset "moments combine exactly" begin
    x = randn(Xoshiro(1), 3, 100); g = randn(Xoshiro(2), 3, 100)
    whole, a, b = CoopMoments(3), CoopMoments(3), CoopMoments(3)
    for j in 1:100
        _fit_moments!(whole, x[:, j], g[:, j])
        _fit_moments!(j <= 40 ? a : b, x[:, j], g[:, j])
    end
    c = _combine_moments!(CoopMoments(3), (a, b), (1.0, 1.0))
    @test c.weight == whole.weight
    @test c.position_mean ≈ whole.position_mean
    @test c.position_m2 ≈ whole.position_m2
    @test c.gradient_mean ≈ whole.gradient_mean
    @test c.gradient_m2 ≈ whole.gradient_m2
    # A weight of 2 counts every state twice.
    twice = _combine_moments!(CoopMoments(3), (whole,), (2.0,))
    @test twice.weight == 200
    @test twice.position_m2 ≈ 2whole.position_m2
    # On a Gaussian the Nutpie scale from positions and gradients is exact.
    σ = [0.1, 1.0, 10.0]
    m = CoopMoments(3)
    rng = Xoshiro(3)
    for _ in 1:1000
        z = σ .* randn(rng, 3)
        _fit_moments!(m, z, -z ./ σ .^ 2)
    end
    @test _moment_scales(m) ≈ σ
end

@testset "recovers the target and stops on target_ess" begin
    out = cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=4, target_ess=1000)
    @test out.stop_reason === :target_ess
    @test out.joint_ess >= 1000
    @test out.n_pooled >= 2
    @test [r.chain_index for r in out.results] == 1:4
    draws = reduce(hcat, [r.posterior_position for r in out.results])
    @test all(abs.(vec(mean(draws; dims=2)) .- target.mu) .< 0.2 .* target.sigma)
    @test all(0.85 .< vec(std(draws; dims=2)) ./ target.sigma .< 1.15)
    for r in out.results
        @test all(isfinite, r.posterior_position)
        @test r.n_samples == size(r.posterior_position, 2)
        @test all(b -> b.batch >= 1, r.batches)
    end
end

@testset "deterministic per seed at any core count" begin
    # A per-chain `n_draws` stop is deterministic; an evaluation budget is a shared
    # counter, so only the round in which it is crossed depends on the schedule.
    run(n_cores) = cooperative_warmup_mcmc(Xoshiro(2), target; n_chains=3, n_cores, n_draws=400)
    one, three = run(1), run(3)
    @test one.stop_reason === three.stop_reason === :n_draws
    for (a, b) in zip(one.results, three.results)
        @test a.posterior_position == b.posterior_position
        @test a.stepsize == b.stepsize
    end
end

@testset "a fitting metric is never restarted" begin
    init = (; position=copy(target.mu), squared_scale=target.sigma .^ 2)
    out = cooperative_warmup_mcmc(Xoshiro(3), target; n_chains=2, init, n_evaluations_budget=40_000)
    @test out.stop_reason === :eval_budget
    @test all(r -> r.n_restarts == 0, out.results)
    @test all(r -> r.n_samples > 0, out.results)
end

@testset "an ill-fitting metric restarts onto the pooled estimate" begin
    init = (; position=copy(target.mu), squared_scale=ones(10))
    out = cooperative_warmup_mcmc(Xoshiro(4), target; n_chains=2, init, target_ess=500)
    @test out.stop_reason === :target_ess
    for r in out.results
        @test r.n_restarts >= 1
        @test all(0.7 .< diag(r.scale) ./ target.sigma .< 1.4)
    end
end

@testset "stops when every chain holds n_draws" begin
    out = cooperative_warmup_mcmc(Xoshiro(5), target; n_chains=3, n_draws=200)
    @test out.stop_reason === :n_draws
    @test all(r -> r.status === :done && size(r.posterior_position, 2) == 200, out.results)
end

@testset "soft clusters keep separated modes apart" begin
    # The scales differ in SHAPE from the unit init metric (a scale proportional to
    # the metric is only a step-size change and restarts nothing), and between the
    # modes, so a pooled estimate across modes would fit neither.
    p = TwoModes([0.5, 1.0, 2.0, 4.0], [4.0, 2.0, 1.0, 0.5])
    start(m) = (; position=fill(m, 4), squared_scale=ones(4))
    init = [start(-MODE), start(-MODE), start(MODE), start(MODE)]
    out = cooperative_warmup_mcmc(Xoshiro(6), p; n_chains=4, init, n_draws=300)
    @test out.stop_reason === :n_draws
    for r in out.results
        sigma = r.chain_index <= 2 ? p.sigma_a : p.sigma_b
        @test all(0.7 .< diag(r.scale) ./ sigma .< 1.4)
    end
end

@testset "API" begin
    # One seed RNG gives every chain its stream (decision 0m8p12j, prong rng).
    @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(1), Xoshiro(2)], target; target_ess=10)
    # Without a finite bound the run would never stop.
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=2)
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=2, target_ess=10,
        init=[(; position=zeros(10), squared_scale=ones(10))])
    # A NamedTuple init's diagonal is the metric until the first batch end; a
    # metric this close lets warm-up finish within the first round.
    out = cooperative_warmup_mcmc(Xoshiro(1), target; n_chains=1, n_draws=1,
        init=(; position=copy(target.mu), squared_scale=4 .* target.sigma .^ 2))
    @test out.stop_reason === :n_draws
    @test only(out.results).n_restarts == 0
    @test diag(only(out.results).scale) ≈ 2 .* target.sigma
end

@testset "checkpoints" begin
    dir = mktempdir()
    out = cooperative_warmup_mcmc(Xoshiro(8), target; n_chains=2, n_evaluations_budget=20_000, checkpoint_dir=dir)
    @test out.stop_reason === :eval_budget
    @test isfile(joinpath(dir, "run_manifest.json")) && isfile(joinpath(dir, "run_summary.json"))
    @test occursin("\"stop_reason\":\"eval_budget\"", read(joinpath(dir, "run_summary.json"), String))
    @test sort(filter(startswith("chain_"), readdir(dir))) == ["chain_1", "chain_2"]
    @test isfile(joinpath(dir, "chain_1", "cp_window_1.jls"))
    p = deserialize(joinpath(dir, "chain_1", "cp_latest.jls"))
    @test p.sampler === :cooperative && p.chain_index == 1
    # The keys the other samplers' readers use (warmuphmc-use §3).
    for k in (:posterior_position, :dropped_posterior_position, :status, :is_final, :stepsize,
              :scale_options, :active_transformation, :position_and_gradient, :reparam_sources,
              :n_divergent_samples, :n_samples, :ess, :total_evaluation_counter, :run_elapsed)
        @test haskey(p, k)
    end
    # A fixed-kernel continuation reads the payload as it reads the other samplers'.
    step = fixed_kernel_nuts_step(FixedKernelNUTSState(p), target)
    @test length(step.state.position) == 10
    # A finished run is not resumed, and a directory holding a run is not reused silently.
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(8), target; n_chains=2,
        n_evaluations_budget=40_000, checkpoint_dir=dir, resume=true)
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(8), target; n_chains=2,
        n_evaluations_budget=20_000, checkpoint_dir=dir)
end

@testset "a resumed run continues as if it had not stopped" begin
    uninterrupted = cooperative_warmup_mcmc(Xoshiro(9), target; n_chains=2, n_draws=300)
    dir = mktempdir()
    # The budget stops the run mid-round; its last checkpoint is the round before.
    cooperative_warmup_mcmc(Xoshiro(9), target; n_chains=2, n_draws=300, n_evaluations_budget=10_000, checkpoint_dir=dir)
    rm(joinpath(dir, "run_summary.json"))   # as if it had crashed before finishing
    @test_throws ArgumentError cooperative_warmup_mcmc(Xoshiro(9), target; n_chains=3, n_draws=300,
        checkpoint_dir=dir, resume=true)
    resumed = cooperative_warmup_mcmc(Xoshiro(123), target; n_chains=2, n_draws=300, checkpoint_dir=dir, resume=true)
    @test resumed.stop_reason === uninterrupted.stop_reason === :n_draws
    for (a, b) in zip(uninterrupted.results, resumed.results)
        @test a.posterior_position == b.posterior_position
        @test a.total_evaluation_counter == b.total_evaluation_counter
    end
    @test resumed.total_evaluation_counter == uninterrupted.total_evaluation_counter
end

@testset "progress tree" begin
    TB = _Treebars
    root = TB.initialize_progress!(:state; description="root")
    out = cooperative_warmup_mcmc(Xoshiro(7), target; n_chains=2, target_ess=300, progress=root)
    text = TB.render_text(root)
    @test occursin("MCMC — target_ess: joint ESS", text)
    for label in ("chains:", "batch:", "pooled ess:", "evaluations:", "elapsed:")
        @test occursin(label, text)
    end
    @test occursin("MCMC.1", text) && occursin("MCMC.2", text)
    @test occursin("divergent samples:", text)
end
end
