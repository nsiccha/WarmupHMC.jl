@testitem "a leapfrog step costs one M⁻¹ product" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LinearAlgebra, LogDensityProblems
    using WarmupHMC: DynamicHMC
    const Pathfinder = WarmupHMC.Pathfinder

    # Snag fixed-kernel-nut-d274a5f0: DynamicHMC applies M⁻¹ = L L' three times
    # per leapfrog step (the half-step momentum, then the new momentum twice, for
    # the leaf's kinetic energy and for its turn statistic), so a step cost six
    # scale products. `TrajectoryEnergy` carries the products along the
    # trajectory: one M⁻¹ product per step.

    # A correlated Gaussian, log p = -x'Px/2 with P = A'A, analytic gradient.
    d = 12
    A = randn(Xoshiro(1), d, d) ./ sqrt(d) + I
    P = Symmetric(A' * A)
    struct Correlated
        P::Symmetric{Float64,Matrix{Float64}}
    end
    LogDensityProblems.capabilities(::Type{Correlated}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(c::Correlated) = size(c.P, 1)
    LogDensityProblems.logdensity(c::Correlated, x) = -dot(x, c.P * x) / 2
    LogDensityProblems.logdensity_and_gradient(c::Correlated, x) = begin
        g = -(c.P * x)
        (dot(x, g) / 2, g)
    end
    lpdf = Correlated(P)

    # Outside its support (x[1] <= 0) this target answers -Inf with a NaN gradient.
    struct HalfLine end
    LogDensityProblems.capabilities(::Type{HalfLine}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(::HalfLine) = 2
    LogDensityProblems.logdensity_and_gradient(::HalfLine, x) =
        x[1] > 0 ? (-sum(abs2, x) / 2, -x) : (-Inf, fill(NaN, 2))

    # Counts M⁻¹ products. A `GaussianKineticEnergy` passed as `metric` is used
    # as given, so its operator is the one the sampler multiplies by.
    struct CountingOperator{M} <: AbstractMatrix{Float64}
        M::M
        n::Base.RefValue{Int}
    end
    Base.size(c::CountingOperator) = size(c.M)
    Base.:*(c::CountingOperator, x::AbstractVector) = (c.n[] += 1; c.M * x)
    counting_energy(L) = DynamicHMC.GaussianKineticEnergy(
        CountingOperator(WarmupHMC.MatrixFactorization(L, L'), Ref(0)),
        WarmupHMC.MatrixInverse(L'))

    # The three scale families the samplers learn.
    pathfinder = WarmupHMC._initial_pathfinder_scale(Pathfinder.WoodburyPDMat(
        Diagonal(fill(0.5, d)), randn(Xoshiro(2), d, 4), Diagonal(fill(0.3, 4))), d)
    @test pathfinder.m1 isa Pathfinder.WoodburyPDLeftFactor
    reflections = WarmupHMC.MatrixFactorization(WarmupHMC.SuccessiveReflections(d), Diagonal(ones(d)))
    positions = cholesky(P).U \ randn(Xoshiro(4), d, 400)
    gradients = -(P * positions)
    WarmupHMC.update_loss!(reflections, positions, gradients)
    @test !isempty(reflections.m1.reflections)
    scales = (; diagonal=Diagonal(fill(0.8, d)), pathfinder, adaptive=reflections)

    @testset "M⁻¹ products per gradient: $name" for (name, L) in pairs(scales)
        state = FixedKernelNUTSState(Xoshiro(5), zeros(d);
            metric=counting_energy(L), stepsize=0.2, max_tree_depth=6)
        products = state.kinetic_energy.M⁻¹.n
        steps = 0
        transitions = 40
        for _ in 1:transitions
            step = fixed_kernel_nuts_step(state, lpdf)
            state = step.state
            steps += step.stats.steps
        end
        # One product per leapfrog step, plus at most three per transition for
        # the vectors it starts from (its momentum, its gradient) and the
        # selected point's final kinetic energy. DynamicHMC's own Hamiltonian
        # takes three per step.
        @test steps < products[] <= steps + 3transitions
        @test products[] < 1.5 * steps
    end

    @testset "carried products equal direct ones to rounding: $name" for (name, L) in pairs(scales)
        κ = WarmupHMC._stream_kinetic_energy(L)
        direct = DynamicHMC.Hamiltonian(κ, lpdf)
        carried = WarmupHMC._hamiltonian(κ, lpdf)
        @test carried.κ isa WarmupHMC.TrajectoryEnergy
        Q = DynamicHMC.evaluate_ℓ(lpdf, randn(Xoshiro(6), d); strict=true)
        p = DynamicHMC.rand_p(Xoshiro(7), κ)
        z_direct = z_carried = DynamicHMC.PhasePoint(Q, p)
        ϵ = 0.1
        for k in 1:30
            ϵk = k <= 20 ? ϵ : -ϵ        # both directions
            z_direct = DynamicHMC.leapfrog(direct, z_direct, ϵk)
            z_carried = DynamicHMC.leapfrog(carried, z_carried, ϵk)
            @test z_carried.Q.q ≈ z_direct.Q.q rtol=1e-10
            @test z_carried.p ≈ z_direct.p rtol=1e-10
            @test DynamicHMC.logdensity(carried, z_carried) ≈
                  DynamicHMC.logdensity(direct, z_direct) rtol=1e-10
            @test DynamicHMC.calculate_p♯(carried, z_carried) ≈
                  κ.M⁻¹ * z_carried.p rtol=1e-10
        end
    end

    @testset "a divergent leaf's gradient is not multiplied" begin
        # The step must neither multiply the NaN gradient nor carry it.
        κ = counting_energy(Diagonal(ones(2)))
        H = WarmupHMC._hamiltonian(κ, HalfLine())
        Q = DynamicHMC.evaluate_ℓ(HalfLine(), [0.05, 0.0]; strict=true)
        z = DynamicHMC.PhasePoint(Q, [-10.0, 0.0])
        n0 = κ.M⁻¹.n[]
        z′ = DynamicHMC.leapfrog(H, z, 0.1)
        @test z′.Q.ℓq == -Inf
        # p♯ and g♯ of the start, and no product of the NaN gradient
        @test κ.M⁻¹.n[] - n0 == 2
        @test H.κ.g[1] === nothing && H.κ.g[2] === nothing
    end
end
