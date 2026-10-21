@testitem "in-memory fixed-kernel NUTS on changing targets" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, LogDensityProblems
    using DifferentiationInterface, Enzyme

    initial = DiagGaussian([0.3, -1.0], [0.5, 1.2])
    shifted = DiagGaussian([2.3, 0.5], [0.8, 1.5])
    q0 = zeros(2)
    kernel = (; metric=initial.sigma, stepsize=0.5, max_tree_depth=8)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)  # exact comparison with the same stream kernel
    try
        scales = copy(initial.sigma)
        state0 = FixedKernelNUTSState(Xoshiro(123), q0;
            metric=scales, stepsize=kernel.stepsize, max_tree_depth=kernel.max_tree_depth)
        q0[1] = 99.0
        scales[1] = 99.0
        @test state0.position == zeros(2)  # caller's start is not retained by alias
        rng0 = WarmupHMC._xoshiro_words(state0.rng)

        path = tempname()
        stream1 = stream_mcmc(Xoshiro(123), initial, zeros(2);
            path, n_draws=1, kernel...)
        first = fixed_kernel_nuts_step(state0, initial)
        @test first.state.position == stream1.position
        @test WarmupHMC._xoshiro_words(first.state.rng) ==
              WarmupHMC._xoshiro_words(stream1.rng)
        @test first.divergent == Bool(stream1.divergences[1])
        @test state0.position == zeros(2)
        @test WarmupHMC._xoshiro_words(state0.rng) == rng0

        stream2 = stream_mcmc(shifted;
            path, n_draws=2, kernel...)
        second = fixed_kernel_nuts_step(first.state, shifted)
        @test second.state.position == stream2.position
        @test WarmupHMC._xoshiro_words(second.state.rng) ==
              WarmupHMC._xoshiro_words(stream2.rng)
        @test second.divergent == Bool(stream2.divergences[2])
        @test first.state.position == stream1.position

        # A third target switch still starts at the last position with a fresh
        # evaluation, matching the file-backed changed-target contract.
        stream3 = stream_mcmc(initial;
            path, n_draws=3, kernel...)
        third = fixed_kernel_nuts_step(second.state, initial)
        @test third.state.position == stream3.position
        @test_throws DimensionMismatch fixed_kernel_nuts_step(
            third.state, DiagGaussian(zeros(3), ones(3)))

        cp_dir = mktempdir()
        adaptive_warmup_mcmc(Xoshiro(7), initial;
            checkpoint_dir=cp_dir, n_draws=40, progress=nothing)
        cp = joinpath(cp_dir, "cp_latest.jls")
        from_checkpoint = FixedKernelNUTSState(cp)
        seeded_stream = stream_mcmc(cp, shifted; path=tempname(), n_draws=1)
        seeded_step = fixed_kernel_nuts_step(from_checkpoint, shifted)
        @test seeded_step.state.position == seeded_stream.position
        @test WarmupHMC._xoshiro_words(seeded_step.state.rng) ==
              WarmupHMC._xoshiro_words(seeded_stream.rng)

        ir = IndexedReparametrization([
            2 => Reparametrization(PartiallyCentered(0.0),
                PartiallyCentered(0.0), 0.0, log(2.0))
        ])
        reparam_target = ReparametrizedProblem(ir, initial, AutoEnzyme())
        sources_before = WarmupHMC.reparam_sources(reparam_target)
        payload = WarmupHMC.deserialize(cp)
        adapted = merge(payload,
            (; reparam_sources=[2 => PartiallyCentered(1.0)]))
        reparam_state = FixedKernelNUTSState(adapted)
        reparam_stream = stream_mcmc(deepcopy(adapted), deepcopy(reparam_target);
            path=tempname(), n_draws=1)
        reparam_step = fixed_kernel_nuts_step(reparam_state, reparam_target)
        @test reparam_step.state.position == reparam_stream.position
        @test WarmupHMC.reparam_sources(reparam_target) == sources_before
        @test sources_before != adapted.reparam_sources

        # The default vector metric stays linear in dimension even after the
        # state snapshots it; a dense 2000×2000 matrix alone is 32 MB.
        large = FixedKernelNUTSState(Xoshiro(8), zeros(2000);
            metric=ones(2000), stepsize=0.5)
        @test Base.summarysize(large) < 1_600_000
    finally
        BLAS.set_num_threads(blas_threads)
    end
end

@testitem "fixed-kernel metric: a dense matrix scale" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra

    # A plain `Matrix` scale used to construct fine and then fail the first step
    # with `MethodError: no method matching ldiv!(::Adjoint{Float64, Matrix{Float64}}, ...)`
    # (snag fixedkernelnutss-dac38aa5); its in-place `L * L'` product also
    # silently returned zeros. The documented contract accepts any matrix scale.
    target = DiagGaussian([0.3, -1.0, 0.5], [0.5, 1.2, 0.8])
    L = LowerTriangular([1.0 0 0; 0.5 1.0 0; 0.2 0.1 1.0])
    step(metric) = fixed_kernel_nuts_step(
        FixedKernelNUTSState(Xoshiro(1), zeros(3); metric, stepsize=0.5), target)
    same_step(a, b) = a.state.position == b.state.position &&
        WarmupHMC._xoshiro_words(a.state.rng) == WarmupHMC._xoshiro_words(b.state.rng)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        # Triangular entries keep the caller's own factor: the step is the one
        # the explicitly wrapped matrix takes, draw for draw.
        @test same_step(step(Matrix(L)), step(L))
        U = UpperTriangular(Matrix(L'))
        @test same_step(step(Matrix(U)), step(U))

        # Any other scale samples through the Cholesky factor of `M * M'`, the
        # only quantity the kernel depends on.
        Q = Matrix(qr(randn(Xoshiro(3), 3, 3)).Q)
        M = Matrix(L) * Q
        @test !istril(M) && !istriu(M)
        @test same_step(step(M), step(cholesky(Symmetric(M * M')).L))
        x = [1.0, -2.0, 0.5]
        inverse_mass_times(metric) = mul!(zeros(3), FixedKernelNUTSState(Xoshiro(1),
            zeros(3); metric, stepsize=0.5).kinetic_energy.M⁻¹, x)
        @test inverse_mass_times(M) ≈ M * M' * x
        S = Symmetric([2.0 0.3 0.1; 0.3 1.5 0.2; 0.1 0.2 1.0])
        @test inverse_mass_times(S) ≈ S * S' * x
        @test all(isfinite, step(S).state.position)

        # The state does not alias the caller's matrix.
        dense = Matrix(L)
        state = FixedKernelNUTSState(Xoshiro(1), zeros(3); metric=dense, stepsize=0.5)
        dense .= 99.0
        @test same_step(fixed_kernel_nuts_step(state, target), step(L))

        # `stream_mcmc` normalizes `metric` the same way.
        dense_stream = stream_mcmc(Xoshiro(1), target, zeros(3);
            path=tempname(), n_draws=2, metric=Matrix(L), stepsize=0.5)
        wrapped_stream = stream_mcmc(Xoshiro(1), target, zeros(3);
            path=tempname(), n_draws=2, metric=L, stepsize=0.5)
        @test dense_stream.draws == wrapped_stream.draws
        close_stream(dense_stream)
        close_stream(wrapped_stream)

        # refused: a scale `L` is square, as its solve `L' \ z` requires
        @test_throws DimensionMismatch FixedKernelNUTSState(Xoshiro(1), zeros(3);
            metric=ones(3, 2), stepsize=0.5)
    finally
        BLAS.set_num_threads(blas_threads)
    end
end
