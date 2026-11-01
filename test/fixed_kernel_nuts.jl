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

@testitem "fixed-kernel particles share one kernel" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra
    using DifferentiationInterface, Enzyme

    d = 40
    target = DiagGaussian(collect(range(-1.0, 1.0; length=d)), collect(range(0.5, 2.0; length=d)))
    reflected(d, k; rng=Xoshiro(1)) = scale_from_data((; diagonal=rand(rng, d) .+ 0.5,
        indices=[collect(1:d) for _ in 1:k],
        reflections=[normalize(randn(rng, d)) for _ in 1:k]))
    # Every family a pilot returns: a Pathfinder run's low-rank factor, the
    # reflections (the structure whose copy costs the most per particle) and a
    # diagonal.
    pilot = adaptive_warmup_mcmc(Xoshiro(11), target; n_draws=100, progress=nothing)
    @test !(pilot.scale_options.pathfinder.m1 isa Diagonal)
    scales = (; pathfinder=pilot.scale_options.pathfinder, reflections=reflected(d, 8),
        diagonal=Diagonal(target.sigma))
    same_step(a, b) = a.state.position == b.state.position &&
        WarmupHMC._xoshiro_words(a.state.rng) == WarmupHMC._xoshiro_words(b.state.rng) &&
        a.stats.steps == b.stats.steps && a.divergent == b.divergent
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        @testset "$family scale" for (family, L) in pairs(scales)
            base = FixedKernelNUTSState(Xoshiro(1), zeros(d);
                metric=L, stepsize=0.3, max_tree_depth=7)
            q = randn(Xoshiro(2), d)
            rng = Xoshiro(3)
            words = WarmupHMC._xoshiro_words(rng)
            particle = FixedKernelNUTSState(base; position=q, rng)

            # One kernel, its own position and stream.
            @test particle.kinetic_energy === base.kinetic_energy
            @test particle.reparam_sources === base.reparam_sources
            @test particle.algorithm === base.algorithm
            @test particle.stepsize === base.stepsize
            @test particle.position == q && particle.position !== q
            @test particle.rng !== rng
            q_copy = copy(q)
            q[1] = 99.0
            @test particle.position == q_copy
            q[1] = q_copy[1]

            # Steps are the ones a state with its own copy of the kernel takes.
            copied = FixedKernelNUTSState(Xoshiro(3), q; metric=L, stepsize=0.3, max_tree_depth=7)
            @test copied.kinetic_energy !== base.kinetic_energy
            a, b = particle, copied
            for _ in 1:5
                sa, sb = fixed_kernel_nuts_step(a, target), fixed_kernel_nuts_step(b, target)
                @test same_step(sa, sb)
                a, b = sa.state, sb.state
            end
            @test a.kinetic_energy === base.kinetic_energy   # stepping keeps sharing
            @test WarmupHMC._xoshiro_words(rng) == words     # the caller's stream did not move

            # Concurrent transitions on the shared kernel equal the serial ones;
            # nothing in the kernel is scratch space.
            n = 16
            particles = [FixedKernelNUTSState(base; position=randn(Xoshiro(i), d), rng=Xoshiro(i))
                for i in 1:n]
            serial = [fixed_kernel_nuts_step(p, target) for p in particles]
            concurrent = Vector{Any}(undef, n)
            Threads.@threads for i in 1:n
                concurrent[i] = fixed_kernel_nuts_step(particles[i], target)
            end
            @test all(same_step.(serial, concurrent))

            # The defaults keep the position and continue a copy of the stream,
            # so a particle that keeps the default rng repeats `base`'s momenta.
            clone = FixedKernelNUTSState(base)
            @test clone.position == base.position && clone.position !== base.position
            @test clone.rng !== base.rng
            @test same_step(fixed_kernel_nuts_step(clone, target), fixed_kernel_nuts_step(base, target))

            # Refused, as for the other constructors.
            @test_throws DimensionMismatch FixedKernelNUTSState(base; position=zeros(d + 1))
            bad = zeros(d)
            bad[2] = NaN
            @test_throws ArgumentError FixedKernelNUTSState(base; position=bad)
        end

        # What a particle holds beyond its position does not grow with the kernel:
        # bytes per particle, from the growth of a vector of particles (so the
        # one kernel they share is not charged to them).
        per_particle(make, n=8) =
            (Base.summarysize([make(i) for i in 1:2n]) - Base.summarysize([make(i) for i in 1:n])) / n
        @testset "memory at d = $D with $k reflections" for (D, k) in ((40, 0), (40, 8), (2000, 16))
            L = reflected(D, k)
            base = FixedKernelNUTSState(Xoshiro(1), zeros(D); metric=L, stepsize=0.3)
            shared = per_particle(i -> FixedKernelNUTSState(base; position=zeros(D), rng=Xoshiro(i)))
            copied = per_particle(i -> FixedKernelNUTSState(Xoshiro(i), zeros(D); metric=L, stepsize=0.3))
            @test shared < Base.summarysize(zeros(D)) + 2048
            # A copied kernel charges each state with the whole operator.
            @test copied - shared >= k * D * sizeof(Float64)
        end

        # A checkpoint state's particles keep its reparametrization controls: each
        # step restores them onto a private copy of the target.
        cp_dir = mktempdir()
        small = DiagGaussian([0.3, -1.0], [0.5, 1.2])
        adaptive_warmup_mcmc(Xoshiro(7), small; checkpoint_dir=cp_dir, n_draws=40, progress=nothing)
        payload = WarmupHMC.deserialize(joinpath(cp_dir, "cp_latest.jls"))
        adapted = merge(payload, (; reparam_sources=[2 => PartiallyCentered(1.0)]))
        ir = IndexedReparametrization([
            2 => Reparametrization(PartiallyCentered(0.0), PartiallyCentered(0.0), 0.0, log(2.0))
        ])
        reparam_target = ReparametrizedProblem(ir, small, AutoEnzyme())
        sources_before = WarmupHMC.reparam_sources(reparam_target)
        from_checkpoint = FixedKernelNUTSState(adapted)
        particle = FixedKernelNUTSState(from_checkpoint; position=[0.1, 0.2], rng=Xoshiro(5))
        seeded = FixedKernelNUTSState(adapted; position=[0.1, 0.2], rng=Xoshiro(5))
        @test particle.kinetic_energy === from_checkpoint.kinetic_energy
        @test particle.reparam_sources === from_checkpoint.reparam_sources
        @test same_step(fixed_kernel_nuts_step(particle, reparam_target),
            fixed_kernel_nuts_step(seeded, reparam_target))
        @test WarmupHMC.reparam_sources(reparam_target) == sources_before
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
