# End-to-end test: adaptive_warmup_mcmc recovers a known posterior.
#
# Runs the full windowed adaptive NUTS warm-up + sampling pipeline (Pathfinder
# init → DynamicHMC NUTS → the three learned linear transformations) against a
# diagonal Gaussian with a non-trivial mean/scale, and checks the posterior
# sample mean/std land within tolerance of the truth.

@testset "single-chain Gaussian recovery" begin
    n = 3
    mu = [1.0, -2.0, 0.5]
    sigma = [1.0, 2.0, 0.5]
    target = DiagGaussian(mu, sigma)
    rng = Xoshiro(20240716)

    result = adaptive_warmup_mcmc(rng, target; n_draws = 600, progress = nothing)

    # Returned NamedTuple carries the documented fields.
    for f in (:initial_position, :halo_position, :halo_gradient, :posterior_position,
              :posterior_gradient, :ess, :scale_options, :active_transformation,
              :stepsize, :total_evaluation_counter, :n_divergent_samples,
              :position_and_gradient, :scale_changes)
        @test hasproperty(result, f)
    end

    draws = result.posterior_position               # dimension × n_samples
    @test size(draws, 1) == n
    @test size(draws, 2) ≥ 600

    post_mean = vec(mean(draws, dims = 2))
    post_std = vec(std(draws, dims = 2))
    for i in 1:n
        @test post_mean[i] ≈ mu[i] atol = 0.3 * sigma[i]
        @test post_std[i] ≈ sigma[i] rtol = 0.3
    end

    # A well-adapted Gaussian should essentially never diverge.
    @test result.n_divergent_samples ≤ 3
end

@testset "multi-chain dispatch" begin
    n = 2
    target = DiagGaussian([0.5, -1.0], [1.0, 1.5])
    rngs = [Xoshiro(1), Xoshiro(2)]
    results = adaptive_warmup_mcmc(rngs, target; n_draws = 300, progress = nothing, parallel = false)
    @test results isa AbstractVector
    @test length(results) == 2
    for r in results
        @test size(r.posterior_position, 1) == n
        @test size(r.posterior_position, 2) ≥ 300
    end
end
