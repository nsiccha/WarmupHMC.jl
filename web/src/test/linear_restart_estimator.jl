@testitem "the running transformed linear restart estimator" setup=[Determinism, Targets] tags=[:linear, :sampler] begin
using WarmupHMC, LinearAlgebra, Random
using WarmupHMC: WeightedScaleAdaptation, effective_sample_size, marginal_scales
import OnlineStatsBase

function _weighted_test_leaves()
    leaves = WarmupHMC.NUTSLeaves(2)
    positions = ([0.0, 1.0], [2.0, -1.0], [-1.0, 3.0], [4.0, 2.0])
    gradients = ([1.0, -2.0], [-0.5, 1.0], [2.0, 0.5], [-1.0, -3.0])
    dH = [0.0, -0.2, -10.0, -0.7]
    for i in eachindex(dH)
        WarmupHMC.record_leaf!(leaves, positions[i], gradients[i], dH[i])
    end
    WarmupHMC.finalize_leaf_weights!(leaves, 2)
    (; leaves, positions, gradients)
end

@testset "running transformed marginal-scale adaptation" begin
    @testset "general weighted moments" begin
        a = WeightedScaleAdaptation(2)
        @test isnan.(marginal_scales(a)) |> all
        @test_throws DimensionMismatch OnlineStatsBase.fit!(a, [1.0], [1.0, 2.0])
        @test_throws DimensionMismatch OnlineStatsBase.fit!(a, [1.0, 2.0], [1.0])
        @test_throws ArgumentError OnlineStatsBase.fit!(a, [1.0, 2.0], [1.0, 2.0]; dw=-1)
    end

    @testset "trajectory policies have exact total weights" begin
        (; leaves, positions, gradients) = _weighted_test_leaves()
        scale = Diagonal([2.0, 0.5])
        stepsize = 0.25
        exact_weight2 = sum(abs2, leaves.weights)

        cases = (
            (:nuts_weighted, :unit, 1.0, exact_weight2),
            (:nuts_weighted, :stepsize, stepsize, stepsize^2 * exact_weight2),
            # Leaves 2 and 4 clear the fixed, mathematical eligibility rule.
            (:all_good_leaves, :unit, 2.0, 2.0),
            (:all_good_leaves, :stepsize, 2stepsize, 2stepsize^2),
        )
        for (source, weighting, expected_weight, expected_weight2) in cases
            a = WeightedScaleAdaptation(2)
            fitted = WarmupHMC._fit_transformed_trajectory!(
                a, zeros(2), zeros(2), scale, leaves, source, weighting, stepsize,
            )
            @test fitted ≈ expected_weight
            @test a.weight ≈ expected_weight
            @test a.weight2 ≈ expected_weight2
        end

        # The implementation learns in the current transformed frame. Compare
        # it with an independently accumulated reference, including W₂.
        observed = WeightedScaleAdaptation(2)
        WarmupHMC._fit_transformed_trajectory!(
            observed, zeros(2), zeros(2), scale, leaves,
            :nuts_weighted, :stepsize, stepsize,
        )
        reference = WeightedScaleAdaptation(2)
        for i in eachindex(leaves.weights)
            OnlineStatsBase.fit!(
                reference, scale \ positions[i], scale' * gradients[i];
                dw=stepsize * leaves.weights[i],
            )
        end
        @test observed.position_mean ≈ reference.position_mean
        @test observed.gradient_mean ≈ reference.gradient_mean
        @test observed.position_m2 ≈ reference.position_m2
        @test observed.gradient_m2 ≈ reference.gradient_m2
        @test observed.weight2 ≈ reference.weight2
        @test marginal_scales(observed) ≈ marginal_scales(reference)
    end

    @testset "later observations dominate finite initial evidence" begin
        a = WeightedScaleAdaptation(1)
        OnlineStatsBase.fit!(a, [-3.0], [2.0])
        OnlineStatsBase.fit!(a, [3.0], [-2.0])
        initial_weight = a.weight
        for _ in 1:100
            OnlineStatsBase.fit!(a, [9.0], [1.0])
            OnlineStatsBase.fit!(a, [11.0], [-1.0])
        end
        # This is the exact weighted-mean identity, not a fitted threshold:
        # the finite initial contribution is W_initial / W_total.
        @test a.position_mean[1] ≈ 2000 / 202
        @test initial_weight / a.weight == 2 / 202
        @test effective_sample_size(a) == 202
    end

    @testset "opt-in replacement and diagonal fallback" begin
        @test WarmupHMC._validate_linear_trajectory_weighting(:unit) === :unit
        @test WarmupHMC._validate_linear_trajectory_weighting(:stepsize) === :stepsize
        @test_throws ArgumentError WarmupHMC._validate_linear_trajectory_weighting(:unknown)
        @test WarmupHMC._validate_linear_restart_source(:halo, :unit) === :halo
        @test WarmupHMC._validate_linear_restart_source(
            :nuts_weighted, :stepsize,
        ) === :nuts_weighted
        @test_throws ArgumentError WarmupHMC._validate_linear_restart_source(:unknown, :unit)
        @test_throws ArgumentError WarmupHMC._validate_linear_restart_source(:halo, :stepsize)

        adaptation = WeightedScaleAdaptation(2)
        for (x, g) in (([-2.0, -1.0], [4.0, 0.5]),
                       ([0.0, 1.0], [0.0, -0.5]),
                       ([2.0, 3.0], [-4.0, 0.5]))
            OnlineStatsBase.fit!(adaptation, x, g)
        end
        expected = marginal_scales(adaptation)
        out = zeros(2)
        unused_position = OnlineStatsBase.Group(
            [OnlineStatsBase.Variance(), OnlineStatsBase.Variance()],
        )
        unused_gradient = deepcopy(unused_position)
        WarmupHMC._linear_restart_scales!(
            out, :nuts_weighted, adaptation, unused_position, unused_gradient,
            zeros(2), Diagonal(ones(2)), zeros(2, 0), zeros(2, 0),
        )
        @test out ≈ expected

        scales = (; diagonal=Diagonal([2.0, 3.0]))
        @test WarmupHMC._apply_linear_metric_fallback!(
            scales, [2.0, 3.0], [0.5, 4.0]; enabled=true,
            old_active=:diagonal, new_active=:diagonal, nonlinear_changed=false,
        )
        @test scales.diagonal.diag == [1.0, 12.0]

        for kwargs in (
            (; enabled=false, old_active=:diagonal, new_active=:diagonal,
               nonlinear_changed=false),
            (; enabled=true, old_active=:pathfinder, new_active=:diagonal,
               nonlinear_changed=false),
            (; enabled=true, old_active=:diagonal, new_active=:adaptive,
               nonlinear_changed=false),
            (; enabled=true, old_active=:diagonal, new_active=:diagonal,
               nonlinear_changed=true),
        )
            candidate = (; diagonal=Diagonal([2.0, 3.0]))
            @test !WarmupHMC._apply_linear_metric_fallback!(
                candidate, [2.0, 3.0], [0.5, 4.0]; kwargs...,
            )
            @test candidate.diagonal.diag == [2.0, 3.0]
        end
    end

    @testset "dense-covariance control discriminates linear families" begin
        covariance = [9.0 2.94; 2.94 1.0]
        target = CorrelatedGaussian(covariance)
        n = 64
        angles = 2pi .* (0:(n - 1)) ./ n
        standardized = sqrt(2) .* permutedims(hcat(cos.(angles), sin.(angles)))
        positions = cholesky(Symmetric(covariance)).L * standardized
        gradients = -(target.precision * positions)

        diagonal = Diagonal(ones(2))
        adaptive = WarmupHMC.MatrixFactorization(
            WarmupHMC.SuccessiveReflections(2), Diagonal(ones(2)),
        )
        diagonal_loss = WarmupHMC.update_loss!(diagonal, positions, gradients)
        adaptive_loss = WarmupHMC.update_loss!(adaptive, positions, gradients)

        @test !isempty(adaptive.m1.reflections)
        @test adaptive_loss < diagonal_loss
    end

    @testset "restart applies the same-family diagonal fallback" begin
        target = DiagGaussian([0.5, 2.0])
        init = (; position=zeros(2), squared_scale=Matrix{Float64}(I, 2, 2))
        # Enter the first window in the diagonal family, then stop at its
        # boundary. `variance_cond_target=1` requests that one restart without
        # turning this integration check into an unbounded adaptation run.
        callback = (state, stage) -> begin
            if stage === :init
                state.active_transformation = :diagonal
                state.kinetic_energy = state.energy_options.diagonal
                return false
            end
            true
        end
        result = adaptive_warmup_mcmc(
            Xoshiro(8), target;
            init, n_draws=100, n_evaluations=40,
            stepsize_adaptation_limit=5, variance_cond_target=1.0,
            nonlinear_adapt=false, monitor_ess=false,
            linear_restart_source=:nuts_weighted,
            linear_trajectory_weighting=:stepsize,
            callback,
        )
        @test result.linear_metric_fallbacks == 1
        @test result.active_transformation === :diagonal
    end
end
end
