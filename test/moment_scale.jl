@testitem "Weighted Gaussian moment summary" begin
    using WarmupHMC, Test, LinearAlgebra, Random, Statistics, Treebars

    dense(L) = hcat((L * e for e in eachcol(Matrix{Float64}(I, size(L)...)))...)
    covariance(summary) = (L = dense(scale_from_data(summary.scale)); L * L')
    function empirical(x, w)
        w = w / sum(w)
        μ = x * w
        centered = x .- μ
        (; mean=μ, covariance=centered * Diagonal(w) * centered')
    end

    @testset "unequal probability masses and a correlated moment oracle" begin
        x = [2.0 -2 1 -1; 2 -2 -1 1]
        w = [0.4, 0.4, 0.1, 0.1]
        fit = fit_gaussian_moments(x; weights=w)
        @test keys(fit) == (:mean, :scale, :diagnostics)
        @test keys(fit.scale) == (:diagonal, :indices, :reflections)
        @test keys(fit.diagnostics) == (:reflections, :iterations,
            :relative_residuals, :converged, :effective_sample_size)
        @test fit.mean ≈ zeros(2) atol=1e-15
        @test covariance(fit) ≈ [3.4 3.0; 3.0 3.4] rtol=2e-8
        @test fit.diagnostics.reflections == 1
        @test fit.diagnostics.converged
        @test fit.diagnostics.effective_sample_size ≈ inv(sum(abs2, w))
        diagonal = fit_gaussian_moments(x; weights=w, max_reflections=0)
        @test isempty(diagonal.scale.reflections)
        @test covariance(diagonal) ≈ Diagonal([3.4, 3.4])
        @test norm(covariance(diagonal) - covariance(fit)) > 4

        # A new factor gives products, solves and determinants through public APIs.
        L = scale_from_data(fit.scale)
        rebuilt = scale_from_data(scale_data(L))
        z = [0.3, -0.7]
        @test rebuilt * z == L * z
        @test rebuilt \ z == L \ z
        @test rebuilt \ (rebuilt * z) ≈ z
        @test first(logabsdet(L)) ≈ log(det([3.4 3.0; 3.0 3.4])) / 2
        @test fit == fit_gaussian_moments(x; weights=w)
        Treebars.with_progress(:state; description="Moment fit test") do progress
            observed = fit_gaussian_moments(x; weights=w, progress)
            @test observed == fit
            @test occursin("Moment directions", Treebars.render_text(progress))
        end
    end

    @testset "all directions: rotations, permutations, weights and non-Gaussian draws" begin
        rng = Xoshiro(17)
        q = Matrix(qr(randn(rng, 5, 5)).Q)
        x = q * Diagonal([5.0, 3, 1, 0.6, 0.2]) * randn(rng, 5, 120)
        x[1, :] .+= 0.3 .* abs2.(view(x, 5, :)) # moment fit is not a Gaussianity claim
        x .+= [3, -2, 7, 0, 1]
        w = rand(rng, 120) .^ 2
        oracle = empirical(x, w)
        fit = fit_gaussian_moments(x; weights=w, maxiter=1000)
        @test fit.mean ≈ oracle.mean
        @test fit.diagnostics.converged
        @test covariance(fit) ≈ oracle.covariance rtol=1e-7
        @test all(<=(1000), fit.diagnostics.iterations)

        perm = [3, 5, 1, 4, 2]
        permuted = fit_gaussian_moments(x[perm, end:-1:1]; weights=reverse(w), maxiter=1000)
        @test permuted.mean ≈ oracle.mean[perm]
        @test covariance(permuted) ≈ oracle.covariance[perm, perm] rtol=1e-7
        rotated = fit_gaussian_moments(q * x; weights=w, maxiter=1000)
        @test covariance(rotated) ≈ q * oracle.covariance * q' rtol=1e-7
        rescaled = fit_gaussian_moments(x; weights=1e300 .* w, maxiter=1000)
        @test rescaled.mean ≈ fit.mean
        @test covariance(rescaled) ≈ covariance(fit) rtol=1e-7
        duplicated = fit_gaussian_moments(repeat(x, 1, 2); weights=vcat(w, w), maxiter=1000)
        @test covariance(duplicated) ≈ covariance(fit) rtol=1e-7

        short = fit_gaussian_moments(x; weights=w, max_reflections=1, maxiter=1)
        @test short.mean ≈ oracle.mean
        @test short.diagnostics.iterations == [1]
        @test !short.diagnostics.converged
        @test norm(covariance(short) - oracle.covariance) / norm(oracle.covariance) > 0.1
        R = dense(scale_from_data(merge(short.scale, (; diagonal=ones(5)))))
        @test short.scale.diagonal .^ 2 ≈ diag(R' * oracle.covariance * R)
    end

    @testset "importance-weighted correlated Gaussian clouds" begin
        for seed in (51, 52, 53)
            rng = Xoshiro(seed)
            q = Matrix(qr(randn(rng, 4, 4)).Q)
            A = q * Diagonal([3.0, 1.5, 0.7, 0.25])
            μ = [1, -2, 0, 3]
            # Proposal N(0, 2I), reweighted to N(0, I), then mapped by A.
            z = sqrt(2.0) * randn(rng, 4, 12000)
            w = exp.(-vec(sum(abs2, z; dims=1)) / 4)
            fit = fit_gaussian_moments(A * z .+ μ; weights=w)
            standardized = A \ covariance(fit) / A'
            @test maximum(abs, A \ (fit.mean - μ)) < 0.06
            @test maximum(abs, diag(standardized) .- 1) < 0.08
            @test norm(standardized - I) < 0.12
            @test fit.diagnostics.converged
        end
    end

    @testset "ownership, views, zero weights and numeric range" begin
        x = [1.0 3 4 8; -2 0 5 6]
        w = [0.2, 0.0, 0.5, 0.3]
        x0, w0 = copy(x), copy(w)
        fit = fit_gaussian_moments(view(x, :, :); weights=view(w, :))
        @test x == x0 && w == w0
        expected = covariance(fit)
        x .= -100
        w .= 0
        @test covariance(fit) == expected
        fit.mean .= 0
        fit.scale.diagonal .= 1
        fit.scale.indices[1][1] = 2
        fit.scale.reflections[1] .= 0
        @test x == fill(-100, size(x)) && w == zeros(4)

        baseline = fit_gaussian_moments(x0[:, [1, 3, 4]]; weights=w0[[1, 3, 4]])
        x0[:, 2] .= 1e300
        ignored = fit_gaussian_moments(x0; weights=w0)
        @test ignored.mean ≈ baseline.mean
        @test covariance(ignored) ≈ covariance(baseline)
        huge_weights = fit_gaussian_moments([0.0 2; 3 5]; weights=[floatmax(Float64), floatmax(Float64)], max_reflections=0)
        @test huge_weights.mean == [1, 4]
        @test huge_weights.scale.diagonal ≈ [1, 1] rtol=1e-15
        tiny = fit_gaussian_moments([1e-200 -1e-200]; weights=[0.5, 0.5], max_reflections=0)
        @test tiny.scale.diagonal ≈ [1e-200] rtol=1e-15
        shifted = fit_gaussian_moments([1e12 + 1 1e12 - 1]; weights=[0.5, 0.5])
        @test shifted.mean == [1e12]
        @test shifted.scale.diagonal ≈ [1] rtol=1e-15
        extreme = [0.0 1e200 0 -1e200; 0 0 1e200 -1e200]
        extreme_weights = [1.0, 1e-320, 1e-320, 1e-320]
        rare = fit_gaussian_moments(extreme; weights=extreme_weights)
        @test covariance(rare) ≈ empirical(extreme, extreme_weights).covariance rtol=1e-7

        Random.seed!(318)
        expected_random = rand()
        Random.seed!(318)
        fit_gaussian_moments([2.0 -2 1 -1; 2 -2 -1 1]; weights=ones(4))
        @test rand() == expected_random
    end

    @testset "invalid and degenerate evidence is explicit" begin
        x = [1.0 -1 2 -2; 1 -1 -2 2]
        @test_throws DimensionMismatch fit_gaussian_moments(x; weights=ones(3))
        for w in (zeros(4), [1, 1, -1, 1], [1, 1, NaN, 1], [1, 1, Inf, 1])
            @test_throws ArgumentError fit_gaussian_moments(x; weights=w)
        end
        for value in (NaN, Inf, -Inf)
            bad = copy(x); bad[1, 1] = value
            @test_throws ArgumentError fit_gaussian_moments(bad; weights=ones(4))
        end
        @test_throws ArgumentError fit_gaussian_moments(zeros(0, 3); weights=ones(3))
        @test_throws ArgumentError fit_gaussian_moments(zeros(3, 0); weights=Float64[])
        @test_throws ArgumentError fit_gaussian_moments(x; weights=ones(4), max_reflections=-1)
        @test_throws ArgumentError fit_gaussian_moments(x; weights=ones(4), maxiter=0)
        @test_throws ArgumentError fit_gaussian_moments(ones(2, 4); weights=ones(4))
        @test_throws ArgumentError fit_gaussian_moments(x; weights=[1, 0, 0, 0])
        @test_throws ArgumentError fit_gaussian_moments([1.0 2 3; 1 2 3]; weights=ones(3))
        @test_throws ArgumentError fit_gaussian_moments([floatmax(Float64) -floatmax(Float64)]; weights=ones(2))
    end
end

@testitem "Weighted moment fit has linear storage and bounded work" begin
    using WarmupHMC, Test, Random, LinearAlgebra
    previous_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        measure(d, n) = begin
            x = randn(Xoshiro(d + n), d, n)
            w = ones(n)
            f() = fit_gaussian_moments(x; weights=w, max_reflections=3, maxiter=4)
            result = f()
            bytes = @allocated f()
            (; bytes, retained=Base.summarysize(result), result)
        end
        small = measure(512, 64)
        dimensions = measure(1024, 64)
        draws = measure(512, 128)
        @test 1.5 < dimensions.bytes / small.bytes < 2.5
        @test 1.5 < draws.bytes / small.bytes < 2.5
        @test 1.5 < dimensions.retained / small.retained < 2.5
        @test draws.retained == small.retained
        @test all(<=(4), dimensions.result.diagnostics.iterations)
        @test dimensions.result.diagnostics.reflections == 3
        @test length(dimensions.result.diagnostics.relative_residuals) == 3
        large = measure(4096, 64)
        @test large.bytes < 8 * 4096^2 / 8 # excludes even one dense covariance
        @info "moment fit scaling" small=(small.bytes, small.retained) dimensions=(dimensions.bytes, dimensions.retained) draws=(draws.bytes, draws.retained) large=(large.bytes, large.retained)
    finally
        BLAS.set_num_threads(previous_blas)
    end
end

@testitem "Moment summary rebuild supports Enzyme reference gradients" begin
    using WarmupHMC, Test, LinearAlgebra, DifferentiationInterface, Enzyme
    @test Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt) !== nothing
    fit = fit_gaussian_moments([2.0 -2 1 -1; 2 -2 -1 1]; weights=[0.4, 0.4, 0.1, 0.1])
    L = scale_from_data(fit.scale)
    logdensity(x, L, μ) = -sum(abs2, L \ (x - μ)) / 2 - first(logabsdet(L)) - length(x) * log(2π) / 2
    x = [0.3, -0.7]
    value, gradient = value_and_gradient(logdensity, AutoEnzyme(), x, Constant(L), Constant(fit.mean))
    @test value ≈ logdensity(x, L, fit.mean)
    @test gradient ≈ -(L' \ (L \ (x - fit.mean))) rtol=1e-12
end
