@testitem "candidate scoring plans are online and source invariant" setup=[Targets, ADBackend, Determinism] tags=[:reparametrization] begin
    using WarmupHMC, LinearAlgebra, LogDensityProblems, Random, Statistics
    using WarmupHMC: IndexedReparametrization, Reparametrization,
                     PartiallyCentered, ReparametrizedProblem,
                     CandidateScoringPlan, reparam_rargs, reparam,
                     restore_reparam_sources!, find_reparametrization!

    const K = 3
    const INDICES = collect(2:4)
    const L = Matrix(cholesky(Symmetric([
        1.0 0.65 -0.25
        0.65 1.0 0.35
        -0.25 0.35 1.0
    ])).L)
    const ALPHA = [0.55, 0.85, 1.15]
    const CSTAR = [0.2, 0.6, 0.9]

    factor(h) = Diagonal(exp.(ALPHA .* h)) * L
    function coordinate_matrix(C, c)
        A = zeros(eltype(C), K, K)
        for k in 1:K
            A[k, k] = C[k, k]^c[k]
            for l in 1:k-1
                A[k, l] = c[k] * C[k, l]
            end
        end
        A
    end

    function invariant_plan(; synchronize! = ir -> ir)
        prepare = function (ir, position, gradient)
            c = [value.source.c for (_, value) in ir.pairs]
            C = factor(position[1])
            A = coordinate_matrix(C, c)
            z = LowerTriangular(A) \ position[INDICES]
            h = A' * gradient[INDICES]
            m = [k == 1 ? 0.0 : dot(@view(C[k, 1:k-1]), @view(z[1:k-1]))
                 for k in 1:K]
            [(source = c[k], z = z[k], m = m[k], log_scale = log(C[k, k]), h = h[k])
             for k in 1:K]
        end
        score = function (frame, pair_number, idx, value, candidate)
            idx == INDICES[pair_number] || return nothing
            f = frame[pair_number]
            t = candidate.c
            scale = exp(t * f.log_scale)
            ((t - f.source) * f.log_scale,
             t * f.m + scale * f.z,
             f.h / scale)
        end
        CandidateScoringPlan(prepare, score; synchronize!)
    end

    make_ir(c) = IndexedReparametrization([
        idx => Reparametrization(
            PartiallyCentered(1.0), PartiallyCentered(c[k]), 0.0, 0.0,
        ) for (k, idx) in enumerate(INDICES)
    ])

    @testset "source controls cancel in the prepared tangent/cotangent frame" begin
        plan = invariant_plan()
        rng = Xoshiro(404)
        sources = (
            [0.0, 0.0, 0.0],
            [0.2, 0.9, 0.1],
            [0.8, 0.1, 0.9],
            [1.0, 1.0, 1.0],
        )
        grid = collect(range(0, 1, 11))
        for _ in 1:12
            hdraw = 1.2randn(rng)
            C = factor(hdraw)
            z = randn(rng, K)
            invariant_gradient = randn(rng, K)
            reference = nothing
            for c in sources
                ir = make_ir(c)
                A = coordinate_matrix(C, c)
                position = [hdraw; A * z]
                source_gradient = UpperTriangular(A') \ invariant_gradient
                frame = plan.prepare(ir, position, [0.0; source_gradient])
                scored = [[plan.score(
                    frame, k, INDICES[k], ir.pairs[k].second, PartiallyCentered(t),
                ) for t in grid] for k in 1:K]
                if isnothing(reference)
                    reference = scored
                else
                    # The position-gradient score is source invariant. The first
                    # tuple element is the candidate/source log-Jacobian change;
                    # it is intentionally source-relative and the default
                    # correlation loss gives it zero weight.
                    @test all(all(isapprox(
                        a, b; rtol=2e-13, atol=2e-13,
                    ) for (a, b) in zip(scored[k][j][2:3], reference[k][j][2:3]))
                    for k in 1:K, j in eachindex(grid))
                end
            end
        end
    end

    @testset "finite-difference pullback and literal endpoints" begin
        plan = invariant_plan()
        rng = Xoshiro(405)
        for c in ([0.1, 0.7, 0.3], [0.9, 0.2, 1.0]), _ in 1:8
            hdraw = 1.2randn(rng)
            C = factor(hdraw)
            A = coordinate_matrix(C, c)
            Astar = coordinate_matrix(C, CSTAR)
            z = randn(rng, K)
            u = A * z
            qstar = Astar * z
            logdensity_u = u_ -> begin
                z_ = LowerTriangular(A) \ u_
                -sum(abs2, Astar * z_) / 2 - logdet(LowerTriangular(A))
            end
            source_gradient = map(1:K) do k
                step = cbrt(eps(Float64)) * max(1.0, abs(u[k]))
                up, um = copy(u), copy(u)
                up[k] += step
                um[k] -= step
                (logdensity_u(up) - logdensity_u(um)) / (2step)
            end
            expected_h = Astar' * (-qstar)
            @test A' * source_gradient ≈ expected_h rtol=2e-7 atol=2e-7

            ir = make_ir(c)
            frame = plan.prepare(ir, [hdraw; u], [0.0; source_gradient])
            for k in 1:K
                zero_obs = plan.score(
                    frame, k, INDICES[k], ir.pairs[k].second, PartiallyCentered(0.0),
                )
                one_obs = plan.score(
                    frame, k, INDICES[k], ir.pairs[k].second, PartiallyCentered(1.0),
                )
                @test zero_obs[2] ≈ z[k]
                @test zero_obs[3] ≈ expected_h[k]
                @test one_obs[2] ≈ (C * z)[k]
                @test one_obs[3] ≈ expected_h[k] / C[k, k]
            end
        end
    end

    @testset "default scoring remains the direct scalar update" begin
        rng = Xoshiro(406)
        ir = make_ir([0.3, 0.6, 0.9])
        direct = WarmupHMC.OnlineReparametrizer(
            ir; accumulator=WarmupHMC.WeightedReparametrizationLoss,
        )
        legacy = deepcopy(direct)
        for _ in 1:20
            position = randn(rng, 4)
            gradient = randn(rng, 4)
            weight = rand(rng)
            WarmupHMC.OnlineStatsBase.fit!(
                ir, direct, position, gradient; weight, count=false,
            )
            for ((idx, value), (_, candidates)) in zip(ir.pairs, legacy.pairs)
                WarmupHMC.OnlineStatsBase.fit!(
                    candidates,
                    reparam_rargs(value, position[idx], gradient[idx], position)...;
                    weight,
                    count=false,
                )
            end
        end
        for ((_, left), (_, right)) in zip(direct.pairs, legacy.pairs),
            ((_, l), (_, r)) in zip(left.pairs, right.pairs)
            @test ntuple(i -> getfield(l, i), fieldcount(typeof(l))) ==
                  ntuple(i -> getfield(r, i), fieldcount(typeof(r)))
        end
    end

    @testset "custom plans force online evidence and retain linear state size" begin
        plan = invariant_plan()
        ir = make_ir([0.2, 0.6, 0.9])
        rp = ReparametrizedProblem(
            ir, Funnel(K), AutoForwardDiff(); scoring_plan=plan,
        )
        online = WarmupHMC.NonlinearRecorder(rp; mode=:linear_pool)
        @test WarmupHMC._effective_nonlinear_evidence(online, rp) === :all_good_leaves
        @test sum(length(candidates.pairs) for (_, candidates) in online.online.pairs) == 11K

        leaves = WarmupHMC.NUTSLeaves(4)
        for i in 1:4
            position = [0.1i; randn(Xoshiro(500 + i), K)]
            gradient = -position
            WarmupHMC.record_leaf!(leaves, position, gradient, -0.1i)
        end
        WarmupHMC.finalize_leaf_weights!(leaves, 2)
        WarmupHMC.record_nonlinear!(online, rp, leaves, 0.25)
        @test all(candidates -> all(loss -> loss.weight == 3, last.(candidates.pairs)),
                  last.(online.online.pairs))

        ordinary = ReparametrizedProblem(make_ir([0.2, 0.6, 0.9]), Funnel(K), AutoForwardDiff())
        retained_halo = WarmupHMC.NonlinearRecorder(ordinary; mode=:linear_pool)
        WarmupHMC.record_nonlinear!(retained_halo, ordinary, leaves, 0.25)
        @test all(candidates -> all(loss -> iszero(loss.weight), last.(candidates.pairs)),
                  last.(retained_halo.online.pairs))

        groups = 4
        many_ir = IndexedReparametrization([
            idx => Reparametrization(
                PartiallyCentered(1.0), PartiallyCentered(0.5), 0.0, 0.0,
            ) for idx in 2:(1 + groups * K)
        ])
        many_rp = ReparametrizedProblem(
            many_ir, Funnel(groups * K), AutoForwardDiff();
            scoring_plan=CandidateScoringPlan((args...) -> nothing, (args...) -> nothing),
        )
        many = WarmupHMC.NonlinearRecorder(many_rp; mode=:linear_pool)
        @test sum(length(candidates.pairs) for (_, candidates) in many.online.pairs) ==
              11groups * K

        prepare_calls = Ref(0)
        seen = Tuple{Int,Int}[]
        ordinal_plan = CandidateScoringPlan(
            (ir, position, gradient) -> (prepare_calls[] += 1; nothing),
            (frame, pair_number, idx, value, candidate) -> begin
                push!(seen, (pair_number, idx))
                nothing
            end,
        )
        ordinal_losses = WarmupHMC.OnlineReparametrizer(
            ir; accumulator=WarmupHMC.WeightedReparametrizationLoss,
        )
        WarmupHMC.OnlineStatsBase.fit!(
            ir, ordinal_losses, zeros(4), ones(4); scoring_plan=ordinal_plan,
        )
        @test prepare_calls[] == 1
        @test seen == repeat(collect(enumerate(INDICES)); inner=11)

        @test WarmupHMC._has_custom_candidate_scoring(rp)
        @test isnothing(WarmupHMC._check_candidate_scoring_compatible(
            (; custom_candidate_scoring=true), rp,
        ))
        ordinary = ReparametrizedProblem(make_ir([0.2, 0.6, 0.9]), Funnel(K), AutoForwardDiff())
        @test_throws ArgumentError WarmupHMC._check_candidate_scoring_compatible(
            (; custom_candidate_scoring=true), ordinary,
        )
        @test_throws ArgumentError WarmupHMC._check_candidate_scoring_compatible(
            (; custom_candidate_scoring=false), rp,
        )
    end

    @testset "source synchronization covers construction, winners, batch, and restore" begin
        synchronized = fill(-1.0, K)
        calls = Ref(0)
        sync! = function (ir)
            calls[] += 1
            synchronized .= [value.source.c for (_, value) in ir.pairs]
        end
        plan = CandidateScoringPlan(
            (ir, position, gradient) -> nothing,
            (args...) -> nothing;
            synchronize! = sync!,
        )
        ir = make_ir(fill(1.0, K))
        rp = ReparametrizedProblem(ir, Funnel(K), AutoForwardDiff(); scoring_plan=plan)
        @test synchronized == ones(K)
        @test calls[] == 1

        recorder = WarmupHMC.NonlinearRecorder(rp; mode=:all_good_leaves)
        for (_, candidates) in recorder.online.pairs, (candidate, loss) in candidates.pairs
            loss.weight = 3
            loss.weight2 = 3
            loss.m2_position = 1
            loss.m2_gradient = 1
            loss.co_position_gradient = abs(candidate.c - 0.3)
            loss.groups = 3
        end
        empty_halo = zeros(4, 0)
        pg = WarmupHMC.DynamicHMC.evaluate_ℓ(rp, zeros(4); strict=false)
        find_reparametrization!(rp, recorder, empty_halo, empty_halo, pg)
        @test synchronized ≈ fill(0.3, K)
        @test calls[] == 2

        restore_reparam_sources!(
            rp, [idx => PartiallyCentered(0.7) for idx in INDICES],
        )
        @test synchronized ≈ fill(0.7, K)
        @test calls[] == 3

        find_reparametrization!(rp, empty_halo, empty_halo, pg)
        @test synchronized ≈ fill(0.7, K)
        @test calls[] == 4
    end

    @testset "fixed-frame winner regression" begin
        rng = Xoshiro(407)
        n = 20_000
        m = zeros(K, n)
        z = zeros(K, n)
        h = zeros(K, n)
        log_scale = zeros(K, n)
        for j in 1:n
            hdraw = 1.2randn(rng)
            C = factor(hdraw)
            Astar = coordinate_matrix(C, CSTAR)
            qstar = randn(rng, K)
            z[:, j] .= LowerTriangular(Astar) \ qstar
            h[:, j] .= Astar' * (-qstar)
            for k in 1:K
                m[k, j] = k == 1 ? 0.0 :
                    dot(@view(C[k, 1:k-1]), @view(z[1:k-1, j]))
                log_scale[k, j] = log(C[k, k])
            end
        end
        grid = collect(range(0, 1, 11))
        winners = map(1:K) do k
            scores = map(grid) do t
                scale = exp.(t .* @view(log_scale[k, :]))
                q = t .* @view(m[k, :]) .+ scale .* @view(z[k, :])
                g = @view(h[k, :]) ./ scale
                cor(q, g)
            end
            grid[argmin(scores)]
        end
        @test winners == [0.5, 0.7, 0.9]
        @test all(0 .< winners .< 1)
    end
end
