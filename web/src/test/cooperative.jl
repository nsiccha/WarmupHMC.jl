# Tests for the cooperative multi-chain sampler (src/cooperative_warmup_mcmc.jl):
# the resumable per-window stepper (CooperativeChain / advance_window! / run_chain!),
# the abandon-stuck predicate (is_stuck), and the public entry cooperative_warmup_mcmc.
#
# Runnable standalone (`julia --project test/cooperative.jl`) or via runtests.jl.
using WarmupHMC, LogDensityProblems, LinearAlgebra, Random, Test
using WarmupHMC: cooperative_chain, advance_window!, run_chain!, chain_draws,
    n_chain_draws, is_stuck, CooperativeChain

# A simple correlated Gaussian target with an analytic gradient (order-1 lpdf).
# Same target the stepper was validated against (bit-for-bit vs adaptive_warmup_mcmc).
struct MvNormalLP
    P::Matrix{Float64}   # precision matrix (SPD)
end
LogDensityProblems.dimension(p::MvNormalLP) = size(p.P, 1)
LogDensityProblems.capabilities(::Type{<:MvNormalLP}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::MvNormalLP, x) = -0.5 * dot(x, p.P, x)
LogDensityProblems.logdensity_and_gradient(p::MvNormalLP, x) = (-0.5 * dot(x, p.P, x), -(p.P * x))

function build_target(dim, seed=123)
    rng = Xoshiro(seed)
    L = LowerTriangular(randn(rng, dim, dim)) + Diagonal(fill(2.0, dim))
    Σ = L * L'
    MvNormalLP(inv(Symmetric(Σ)))
end

# A cheap CooperativeChain that skips Pathfinder: passing a NamedTuple `init`
# short-circuits initialize_mcmc, so is_stuck's pure-predicate logic can be
# exercised without paying the variational-init cost.
function cheap_chain(dim=3; kwargs...)
    lpdf = build_target(dim)
    # squared_scale must be dense & non-diagonal: factorize() returns a Diagonal
    # (which has no `.L`) for a structurally-diagonal matrix. Use the target
    # covariance, which is a genuine dense SPD matrix (as Pathfinder would yield).
    sq = Matrix(inv(Symmetric(lpdf.P)))
    cooperative_chain(Xoshiro(1), lpdf;
        init=(; position=zeros(dim), squared_scale=sq),
        nonlinear_adapt=false, kwargs...)
end

# Build a checkpoint log entry matching log_checkpoint!'s schema (only `status`
# and `variance_cond` are read by is_stuck, but keep the shape realistic).
ckpt(; status, variance_cond, n_samples=0) = (;
    window=0, n_samples, min_ess=NaN, variance_cond,
    n_divergent_samples=0, total_evaluation_counter=0, steps_per_draw=NaN, status)

@testset "cooperative sampler" begin

    @testset "is_stuck predicate" begin
        chain = cheap_chain()

        # Too few checkpoints to judge -> not stuck.
        empty!(chain.checkpoints)
        chain.n_samples = 0; chain.n_divergent_samples = 0
        @test is_stuck(chain) == false
        append!(chain.checkpoints, [ckpt(; status=:warming, variance_cond=10.0) for _ in 1:3])
        @test is_stuck(chain) == false   # still < min_windows

        # Geometry stall: 4 warming windows, condition number not improving.
        empty!(chain.checkpoints)
        append!(chain.checkpoints, [ckpt(; status=:warming, variance_cond=10.0) for _ in 1:4])
        @test is_stuck(chain) == true

        # Warming but condition number is improving fast -> not a stall.
        empty!(chain.checkpoints)
        append!(chain.checkpoints, [ckpt(; status=:warming, variance_cond=c) for c in (100.0, 50.0, 20.0, 5.0)])
        @test is_stuck(chain) == false

        # Not all-warming (one reached :sampling) -> geometry branch does not fire.
        empty!(chain.checkpoints)
        append!(chain.checkpoints, [ckpt(; status=:warming, variance_cond=10.0) for _ in 1:3])
        push!(chain.checkpoints, ckpt(; status=:sampling, variance_cond=10.0))
        @test is_stuck(chain) == false

        # Divergence blow-up: > divergence_rate of the current draws diverged.
        empty!(chain.checkpoints)
        append!(chain.checkpoints, [ckpt(; status=:sampling, variance_cond=1.5) for _ in 1:4])
        chain.n_samples = 100; chain.n_divergent_samples = 50   # 0.5 > 0.4
        @test is_stuck(chain) == true

        # Divergent fraction below the threshold -> not stuck.
        chain.n_samples = 100; chain.n_divergent_samples = 20   # 0.2 < 0.4
        @test is_stuck(chain) == false

        # High fraction but too few samples to judge on -> not stuck.
        chain.n_samples = 10; chain.n_divergent_samples = 8      # < min_divergence_samples
        @test is_stuck(chain) == false
    end

    @testset "single-chain reproduction (bit-for-bit vs adaptive_warmup_mcmc)" begin
        # run_chain! on a CooperativeChain must reproduce the monolith exactly,
        # given the same seed and config (the resumable-stepper correctness claim).
        cfg = (; n_draws=200, nonlinear_adapt=false, monitor_ess=false)
        for (dim, seed) in [(3, 1), (5, 42)]
            lpdf = build_target(dim)
            a = adaptive_warmup_mcmc(Xoshiro(seed), lpdf; cfg...)
            chain = run_chain!(cooperative_chain(Xoshiro(seed), lpdf; cfg...))
            b = chain_draws(chain)
            @test size(a.posterior_position) == size(b)
            @test a.posterior_position == b            # exact, maxdiff == 0
            @test a.stepsize == chain.stepsize
            @test a.active_transformation == chain.active_transformation
        end
    end

    @testset "kinetic_energy accepts every transformation" begin
        # `advance_window!` reassigns `chain.kinetic_energy` from `energy_options`
        # at every window boundary, and the three options are heterogeneously
        # typed (each linear transformation yields a distinct
        # GaussianKineticEnergy type). A concretely-typed field made every
        # restart that switched away from the construction-time choice throw
        # `MethodError: Cannot convert`, so the field must stay untyped — same
        # as `AWMState.kinetic_energy`.
        chain = cheap_chain(8)
        opts = chain.energy_options
        @test length(unique(typeof(opts[k]) for k in keys(opts))) == length(keys(opts))
        for k in keys(opts)
            chain.kinetic_energy = opts[k]
            @test chain.kinetic_energy === opts[k]
        end
    end

    @testset "cooperative_warmup_mcmc public entry" begin
        lpdf = build_target(3)

        # At least one finite stopping bound is required.
        @test_throws AssertionError cooperative_warmup_mcmc([Xoshiro(1)], lpdf)

        # Eval-budget-bounded run: pools draws across chains, stops on budget.
        rngs = [Xoshiro(s) for s in 1:3]
        out = cooperative_warmup_mcmc(rngs, lpdf;
            n_cores=2, n_evaluations_budget=8000, nonlinear_adapt=false)
        @test out.n_started >= 1
        @test out.n_used >= 1
        @test out.total_evaluation_counter > 0
        @test isfinite(out.joint_ess)
        @test length(out.results) == out.n_started
        # Pooled posterior mean of a zero-mean Gaussian is near zero.
        used = [r.posterior_position for r in out.results
                if (r.status === :sampling || r.status === :done) && size(r.posterior_position, 2) > 0]
        @test !isempty(used)
        pooled = reduce(hcat, used)
        pooled_mean = vec(sum(pooled; dims=2)) ./ size(pooled, 2)
        @test maximum(abs, pooled_mean) < 0.75

        # Never advances more distinct chains than there are RNGs.
        @test out.n_started <= length(rngs)
    end

end
