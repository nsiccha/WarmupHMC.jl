# Focused contract tests for strict-online candidate scoring. The concrete
# triangular frame here is a synthetic correlated-random-effect control; the
# package API remains model-agnostic.

const ACE_K = 3
const ACE_INDICES = collect(2:4)
const ACE_L = Matrix(cholesky(Symmetric([
    1.0 0.65 -0.25
    0.65 1.0 0.35
    -0.25 0.35 1.0
])).L)
const ACE_ALPHA = [0.55, 0.85, 1.15]
const ACE_CSTAR = [0.2, 0.6, 0.9]

ace_factor(h) = Diagonal(exp.(ACE_ALPHA .* h)) * ACE_L
function ace_coordinate_matrix(C, c)
    A = zeros(eltype(C), ACE_K, ACE_K)
    for k in 1:ACE_K
        A[k, k] = C[k, k]^c[k]
        for l in 1:k-1
            A[k, l] = c[k] * C[k, l]
        end
    end
    A
end

ace_ir(c) = WarmupHMC.IndexedReparametrization([
    idx => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(1.0),
        WarmupHMC.PartiallyCentered(c[k]),
        0.0,
        0.0,
    ) for (k, idx) in enumerate(ACE_INDICES)
])

function ace_plan(; synchronize! = identity)
    prepare = function (ir, position, gradient)
        c = [value.source.c for (_, value) in ir.pairs]
        C = ace_factor(position[1])
        A = ace_coordinate_matrix(C, c)
        z = LowerTriangular(A) \ position[ACE_INDICES]
        h = A' * gradient[ACE_INDICES]
        map(1:ACE_K) do k
            m = k == 1 ? 0.0 : dot(@view(C[k, 1:k-1]), @view(z[1:k-1]))
            (; source=c[k], z=z[k], m, log_scale=log(C[k, k]), h=h[k])
        end
    end
    score = function (frame, pair_number, idx, value, candidate)
        idx == ACE_INDICES[pair_number] || return nothing
        f = frame[pair_number]
        t = candidate.c
        scale = exp(t * f.log_scale)
        ((t - f.source) * f.log_scale,
         t * f.m + scale * f.z,
         f.h / scale)
    end
    WarmupHMC.CandidateScoringPlan(prepare, score; synchronize!)
end

@testset "source controls cancel from position-gradient scores" begin
    plan = ace_plan()
    rng = Xoshiro(620)
    source_vectors = (
        [0.0, 0.0, 0.0], [0.2, 0.9, 0.1],
        [0.8, 0.1, 0.9], [1.0, 1.0, 1.0],
    )
    grid = collect(range(0, 1, 11))
    for _ in 1:8
        hyper = 1.2randn(rng)
        C = ace_factor(hyper)
        z = randn(rng, ACE_K)
        invariant_gradient = randn(rng, ACE_K)
        reference = nothing
        for c in source_vectors
            ir = ace_ir(c)
            A = ace_coordinate_matrix(C, c)
            position = [hyper; A * z]
            source_gradient = UpperTriangular(A') \ invariant_gradient
            frame = plan.prepare(ir, position, [0.0; source_gradient])
            scored = [[plan.score(
                frame, k, ACE_INDICES[k], ir.pairs[k].second,
                WarmupHMC.PartiallyCentered(t),
            )[2:3] for t in grid] for k in 1:ACE_K]
            if isnothing(reference)
                reference = scored
            else
                @test all(all(isapprox(a, b; rtol=2e-13, atol=2e-13)
                              for (a, b) in zip(scored[k][j], reference[k][j]))
                          for k in 1:ACE_K, j in eachindex(grid))
            end
        end
    end
end

@testset "transport gradient and literal endpoints" begin
    plan = ace_plan()
    rng = Xoshiro(621)
    for c in ([0.1, 0.7, 0.3], [0.9, 0.2, 1.0]), _ in 1:5
        hyper = 1.2randn(rng)
        C = ace_factor(hyper)
        A = ace_coordinate_matrix(C, c)
        Astar = ace_coordinate_matrix(C, ACE_CSTAR)
        z = randn(rng, ACE_K)
        u = A * z
        qstar = Astar * z
        logdensity_u = u_ -> begin
            z_ = LowerTriangular(A) \ u_
            -sum(abs2, Astar * z_) / 2 - logdet(LowerTriangular(A))
        end
        source_gradient = fd_gradient(logdensity_u, u)
        h = Astar' * (-qstar)
        @test A' * source_gradient ≈ h rtol=2e-5 atol=2e-5

        ir = ace_ir(c)
        frame = plan.prepare(ir, [hyper; u], [0.0; source_gradient])
        for k in 1:ACE_K
            at_zero = plan.score(
                frame, k, ACE_INDICES[k], ir.pairs[k].second,
                WarmupHMC.PartiallyCentered(0.0),
            )
            at_one = plan.score(
                frame, k, ACE_INDICES[k], ir.pairs[k].second,
                WarmupHMC.PartiallyCentered(1.0),
            )
            @test at_zero[2] ≈ z[k]
            @test at_zero[3] ≈ h[k]
            @test at_one[2] ≈ (C * z)[k]
            @test at_one[3] ≈ h[k] / C[k, k]
        end
    end
end

@testset "one prepare, deterministic ordinals, and linear persistent state" begin
    ir = ace_ir(ACE_CSTAR)
    prepare_calls = Ref(0)
    seen = Tuple{Int,Int}[]
    plan = WarmupHMC.CandidateScoringPlan(
        (ir, position, gradient) -> (prepare_calls[] += 1; nothing),
        (frame, pair_number, idx, value, candidate) -> begin
            push!(seen, (pair_number, idx))
            nothing
        end,
    )
    losses = WarmupHMC.OnlineReparametrizer(
        ir; accumulator=WarmupHMC.WeightedReparametrizationLoss,
    )
    WarmupHMC.OnlineStatsBase.fit!(
        ir, losses, zeros(4), ones(4); scoring_plan=plan,
    )
    @test prepare_calls[] == 1
    @test seen == repeat(collect(enumerate(ACE_INDICES)); inner=11)

    groups = 4
    many_ir = WarmupHMC.IndexedReparametrization([
        idx => WarmupHMC.Reparametrization(
            WarmupHMC.PartiallyCentered(1.0),
            WarmupHMC.PartiallyCentered(0.5), 0.0, 0.0,
        ) for idx in 2:(1 + groups * ACE_K)
    ])
    target = DiagGaussian(zeros(1 + groups * ACE_K), ones(1 + groups * ACE_K))
    rp = WarmupHMC.ReparametrizedProblem(
        many_ir, target, AutoEnzyme(); scoring_plan=plan,
    )
    recorder = WarmupHMC.NonlinearRecorder(rp; mode=:linear_pool)
    @test WarmupHMC._effective_nonlinear_evidence(recorder, rp) === :all_good_leaves
    @test sum(length(candidates.pairs) for (_, candidates) in recorder.online.pairs) ==
          11groups * ACE_K

    ordinary = WarmupHMC.ReparametrizedProblem(ace_ir(ACE_CSTAR),
        DiagGaussian(zeros(4), ones(4)), AutoEnzyme())
    @test_throws ArgumentError WarmupHMC._check_candidate_scoring_compatible(
        (; custom_candidate_scoring=true), ordinary,
    )
    @test_throws ArgumentError WarmupHMC._check_candidate_scoring_compatible(
        (; custom_candidate_scoring=false), rp,
    )
end

@testset "fixed-frame winner regression" begin
    rng = Xoshiro(622)
    n = 20_000
    m = zeros(ACE_K, n)
    z = zeros(ACE_K, n)
    h = zeros(ACE_K, n)
    log_scale = zeros(ACE_K, n)
    for j in 1:n
        hyper = 1.2randn(rng)
        C = ace_factor(hyper)
        Astar = ace_coordinate_matrix(C, ACE_CSTAR)
        qstar = randn(rng, ACE_K)
        z[:, j] .= LowerTriangular(Astar) \ qstar
        h[:, j] .= Astar' * (-qstar)
        for k in 1:ACE_K
            m[k, j] = k == 1 ? 0.0 :
                dot(@view(C[k, 1:k-1]), @view(z[1:k-1, j]))
            log_scale[k, j] = log(C[k, k])
        end
    end
    grid = collect(range(0, 1, 11))
    winners = map(1:ACE_K) do k
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
