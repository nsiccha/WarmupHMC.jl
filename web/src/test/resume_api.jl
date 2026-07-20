# Tests for the unified resume API: `checkpoint_dir` + `resume=` / `overwrite=` on
# the samplers themselves, and the checkpoint payload's reduction to PURE SAMPLER
# STATE (config now comes from the resuming call, which is what makes
# resume-and-extend possible).
#
# Runnable standalone (`julia --project test/resume_api.jl`) or via runtests.jl.
using WarmupHMC, LogDensityProblems, LinearAlgebra, Random, Test
using Serialization: deserialize

struct MvNormalLP; P::Matrix{Float64}; end
LogDensityProblems.dimension(p::MvNormalLP) = size(p.P, 1)
LogDensityProblems.capabilities(::Type{<:MvNormalLP}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::MvNormalLP, x) = -0.5 * dot(x, p.P, x)
LogDensityProblems.logdensity_and_gradient(p::MvNormalLP, x) = (-0.5*dot(x,p.P,x), -(p.P*x))
function build_target(dim, seed=123)
    rng = Xoshiro(seed); L = LowerTriangular(randn(rng, dim, dim)) + Diagonal(fill(2.0, dim))
    MvNormalLP(inv(Symmetric(L*L')))
end

lpdf = build_target(3)
# An explicit `init` NamedTuple short-circuits `initialize_mcmc`, so these tests
# never touch Pathfinder (whose default AutoForwardDiff backend is unavailable in
# this environment — see todo 1hb814i). Same trick the cooperative tests use.
_init(l) = (; position=zeros(LogDensityProblems.dimension(l)),
             squared_scale=Matrix(inv(Symmetric(l.P))))
cfg = (; n_draws=200, nonlinear_adapt=false, monitor_ess=false, init=_init(lpdf))
mktempdir() do d
    @testset "resume=/overwrite= on the sampler" begin
        # 1. fresh run into an empty dir needs no flag
        a = adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
        @test size(a.posterior_position, 2) >= 200

        # 2. payload is pure state: no config, no dead `stage`
        p = deserialize(joinpath(d, "cp_latest.jls"))
        for k in (:n_draws, :stepsize_adaptation_limit, :variance_cond_target,
                  :nonlinear_adapt, :monitor_ess, :recording_target, :kwargs,
                  :algorithm, :stepsize_adaptation, :stage)
            @test !hasproperty(p, k)
        end
        @test p.sampler === :adaptive
        @test p.schema_version == 2
        @test hasproperty(p, :position_and_gradient)   # state still there

        # 3. re-running into a non-empty dir refuses to guess
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf;
            checkpoint_dir=d, resume=true, overwrite=true, cfg...)

        # 4. RESUME-AND-EXTEND: config comes from the call, so a larger n_draws keeps going
        b = adaptive_warmup_mcmc(Xoshiro(1), lpdf;
            checkpoint_dir=d, resume=true, n_draws=500, nonlinear_adapt=false,
            monitor_ess=false, init=_init(lpdf))
        @test size(b.posterior_position, 2) >= 500

        # 5. overwrite starts fresh
        c = adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, overwrite=true, cfg...)
        @test size(c.posterior_position, 2) >= 200
        # ...and reproduces the original run exactly
        @test c.posterior_position == a.posterior_position

        # 6. flags without a dir are meaningless
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; resume=true, cfg...)
    end
end

mktempdir() do d
    @testset "masquerade guard + validation" begin
        adaptive_warmup_mcmc(Xoshiro(1), lpdf; checkpoint_dir=d, cfg...)
        path = joinpath(d, "cp_latest.jls")
        # dimension mismatch is caught
        @test_throws DimensionMismatch WarmupHMC.restore_state(
            deserialize(path), build_target(5), nothing)
        # a cooperative-tagged payload is refused by the adaptive reader
        p = deserialize(path)
        coop = merge(NamedTuple(pairs(p)), (; sampler=:cooperative))
        @test_throws ArgumentError WarmupHMC.restore_state(coop, lpdf, nothing)
        # pre-tag checkpoints (no `sampler` field) still read as adaptive
        @test WarmupHMC.checkpoint_sampler((; a=1)) === :adaptive
        # recording_target cannot change on resume
        @test_throws ArgumentError WarmupHMC.restore_state(p, lpdf, nothing; recording_target=7)
    end
end

mktempdir() do d
    @testset "cooperative run-dir guard" begin
        cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
            n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, init=_init(lpdf))
        @test !isempty(filter(f -> startswith(f, "chain_"), readdir(d)))
        # a second run into the same dir would interleave — refuse
        @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
            n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, init=_init(lpdf))
        # resume is honestly reported as unsupported, not silently ignored
        @test_throws ArgumentError cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
            n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, resume=true, init=_init(lpdf))
        # overwrite clears it
        cooperative_warmup_mcmc([Xoshiro(s) for s in 1:2], lpdf;
            n_cores=2, n_evaluations_budget=6000, nonlinear_adapt=false, checkpoint_dir=d, overwrite=true, init=_init(lpdf))
    end
end
