@testitem "NUTS leaf weights and recording" setup=[Determinism] tags=[:recording] begin
    # `Random` was never declared here before the TestItemRunner migration — this
    # file reached `Xoshiro` through `runtests.jl`'s own `using Test, WarmupHMC,
    # Random, LinearAlgebra`, which every `include`d file inherited. Item modules
    # do not inherit anything, so the dependency had to become explicit.
    using WarmupHMC, Random

    const DynamicHMC = WarmupHMC.DynamicHMC
    const LogDensityProblems = WarmupHMC.LogDensityProblems

    function brute_force_leaf_weights(dH, depth)
        weights = zeros(length(dH))
        weights[1] = 1
        previous_mass = exp(dH[1])
        first_new = 2
        for level in 0:(depth - 1)
            n_new = 1 << level
            new_idxs = first_new:(first_new + n_new - 1)
            new_masses = exp.(dH[new_idxs])
            new_mass = sum(new_masses)
            switch_probability = min(new_mass / previous_mass, 1)
            weights[1:(first_new - 1)] .*= 1 - switch_probability
            weights[new_idxs] .= switch_probability .* new_masses ./ new_mass
            previous_mass += new_mass
            first_new += n_new
        end
        weights
    end

    # Exact reference trajectory: proposals carry their whole probability vector
    # instead of drawing one state. This lets DynamicHMC's own tree combinator
    # brute-force the marginal distribution for every direction pattern.
    mutable struct MarginalTrajectory{F}
        logweight::F
        visited::Vector{Int}
    end
    DynamicHMC.move(::MarginalTrajectory, z, is_forward) = z + (is_forward ? 1 : -1)
    DynamicHMC.combine_visited_statistics(::MarginalTrajectory, ::Nothing, ::Nothing) = nothing
    DynamicHMC.combine_turn_statistics(::MarginalTrajectory, ::Nothing, ::Nothing) = nothing
    DynamicHMC.is_turning(::MarginalTrajectory, ::Nothing) = false
    DynamicHMC.calculate_logprob2(::MarginalTrajectory, is_doubling, w1, w2, w) =
        DynamicHMC.biased_progressive_logprob2(is_doubling, w1, w2, w)
    function DynamicHMC.leaf(trajectory::MarginalTrajectory, z, is_initial)
        push!(trajectory.visited, z)
        ((z:z, [0.0]), trajectory.logweight(z), nothing), nothing
    end
    function DynamicHMC.combine_proposals(
        _, ::MarginalTrajectory, proposal1, proposal2, logprob2, is_forward,
    )
        probability2 = min(exp(logprob2), 1.0)
        probability1 = 1 - probability2
        if !is_forward
            proposal1, proposal2 = proposal2, proposal1
            probability1, probability2 = probability2, probability1
        end
        positions1, weights1 = proposal1
        positions2, weights2 = proposal2
        first(positions2) == last(positions1) + 1 || error("non-adjacent reference proposals")
        first(positions1):last(positions2),
            vcat(probability1 .* exp.(weights1), probability2 .* exp.(weights2)) .|> log
    end

    @testset "NUTS leaf proposal weights" begin
        cases = (
            ([0.0], 0),
            ([0.0, -2.0], 1),
            ([0.0, -1.5, -0.2, -2.1], 2),
            ([0.0, -3.0, 0.2, -1.0, -0.4, -2.3, 0.7, -1.8], 3),
        )
        for (dH, depth) in cases
            weights = WarmupHMC.leaf_weights!(Float64[], dH, depth)
            @test sum(weights) ≈ 1
            @test weights ≈ brute_force_leaf_weights(dH, depth)
        end

        # Leaves visited in an invalid final doubling are excluded from the
        # successfully sampled tree and must carry no proposal probability.
        dH = [0.0, -0.5, -0.2, -1.0, 3.0, 4.0]
        weights = WarmupHMC.leaf_weights!(Float64[], dH, 2)
        @test sum(weights) ≈ 1
        @test weights[5:6] == [0.0, 0.0]
        @test weights ≈ brute_force_leaf_weights(dH, 2)

        logweight(z) = -0.13 * (z - 1)^2 + 0.07 * z
        for depth in 1:3, flags in 0:(1 << depth) - 1
            trajectory = MarginalTrajectory(logweight, Int[])
            proposal, _, _, sampled_depth = DynamicHMC.sample_trajectory(
                nothing, trajectory, 0, depth, DynamicHMC.Directions(UInt32(flags)),
            )
            @test sampled_depth == depth
            positions, logprobabilities = proposal
            weights = WarmupHMC.leaf_weights!(
                Float64[], logweight.(trajectory.visited), sampled_depth,
            )
            by_position = Dict(position => weight for (position, weight) in zip(trajectory.visited, weights))
            @test [by_position[position] for position in positions] ≈ exp.(logprobabilities)
        end
    end

    @testset "expected leaf statistics" begin
        leaves = WarmupHMC.NUTSLeaves(2)
        positions = ([0.0, 0.0], [1.0, 2.0], [-2.0, 1.0], [3.0, -1.0])
        gradients = ([0.0, 0.0], [-1.0, -2.0], [2.0, -1.0], [-3.0, 1.0])
        dH = [0.0, -0.7, -0.1, -1.2]
        for i in eachindex(dH)
            WarmupHMC.record_leaf!(leaves, positions[i], gradients[i], dH[i])
        end
        WarmupHMC.finalize_leaf_weights!(leaves, 2)

        expected_position = sum(eachindex(dH)) do i
            leaves.weights[i] .* positions[i]
        end
        @test WarmupHMC.expected_stat(s -> s.position, leaves) ≈ expected_position
        @test WarmupHMC.expected_stat(s -> sum(s.position .* s.gradient), leaves) ≈
              sum(i -> leaves.weights[i] * sum(positions[i] .* gradients[i]), eachindex(dH))
    end

    struct StandardNormalLogDensity end
    LogDensityProblems.capabilities(::Type{StandardNormalLogDensity}) =
        LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(::StandardNormalLogDensity) = 1
    LogDensityProblems.logdensity(::StandardNormalLogDensity, q) = -sum(abs2, q) / 2
    LogDensityProblems.logdensity_and_gradient(::StandardNormalLogDensity, q) =
        (-sum(abs2, q) / 2, -q)

    @testset "DynamicHMC-native leaf recording" begin
        rng = Random.Xoshiro(42)
        recording = WarmupHMC.RecordingPosterior2(
            StandardNormalLogDensity(); rng, recorder=WarmupHMC.LimitedRecorder2(8),
        )
        kinetic = DynamicHMC.GaussianKineticEnergy(1)
        hamiltonian = DynamicHMC.Hamiltonian(kinetic, recording)
        initial = DynamicHMC.evaluate_ℓ(recording, [0.25]; strict=true)

        WarmupHMC.reset!(recording.leaves)
        _, stats = DynamicHMC.sample_tree(
            rng, DynamicHMC.NUTS(max_depth=3), hamiltonian, initial, 0.15;
            p=[0.5], directions=DynamicHMC.Directions(UInt32(0b101)),
        )
        WarmupHMC.finalize_leaf_recording!(recording, stats.depth)

        @test length(recording.leaves) == stats.steps + 1
        @test sum(recording.leaves.weights) ≈ 1
        @test all(iszero, recording.leaves.weights[(1 << stats.depth) + 1:end])
        # The halo must NOT be reduced to one state per trajectory: `record!` retains
        # one state per `thin` leaf evaluations during the traversal. With `thin=1`
        # that is every leaf whose Hamiltonian error clears `log(1e-2)`, capped by
        # the ring. Regressing this to a single draw per trajectory shrinks the pool
        # by the mean tree size and starves both halo consumers.
        n_eligible = count(2:length(recording.leaves)) do i
            recording.leaves.dH[i] > log(1e-2)
        end
        @test size(recording.halo_position, 2) == min(n_eligible, recording.recorder.target)
        @test size(recording.halo_position, 2) > 1
        # Every retained halo column is one of the leaves actually visited.
        @test all(axes(recording.halo_position, 2)) do col
            any(axes(recording.leaves.position, 2)) do i
                recording.halo_position[:, col] == recording.leaves.position[:, i] &&
                    recording.halo_gradient[:, col] == recording.leaves.gradient[:, i]
            end
        end
    end
end
