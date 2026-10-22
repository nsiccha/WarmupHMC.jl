@testitem "MatrixExpressions" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Pkg, TOML
    using Distributions
    using DifferentiationInterface, Enzyme

# Unit tests for src/MatrixExpressions.jl — matrix-free operators.
#
# These types subtype `AbstractMatrix` for the operator interface
# (`mul!`/`ldiv!`/`adjoint`/`size`) but deliberately define NO `getindex`, so
# any element-indexing op throws. We test the matrix-free methods against dense
# references, plus a regression for the structural `hash`/`isequal`/`==` added
# in commit `3b3c05d` (hashing an operator as a Dict key used to crash).

const MI = WarmupHMC.MatrixInverse
const MF = WarmupHMC.MatrixFactorization
const SR = WarmupHMC.SuccessiveReflections
const STR = WarmupHMC.ScaleThenReflect

# A well-conditioned lower-triangular factor (realistic: a Cholesky L).
_lower(n) = LowerTriangular(randn(n, n) + 2n * I)
_diag(n) = Diagonal(rand(n) .+ 0.5)
# A single normalized Householder reflection over all n indices.
_reflection(v) = SR([collect(1:length(v))], [v], zeros(length(v)), zeros(length(v)), zeros(length(v)))

@testset "MatrixInverse" begin
    n = 4
    x = randn(n); y = similar(x)
    for M in (_diag(n), _lower(n))
        A = MI(M)
        @test size(A) == size(M)
        @test size(A, 1) == n
        mul!(y, A, x);  @test y ≈ M \ x        # mul! solves
        ldiv!(y, A, x); @test y ≈ M * x        # ldiv! multiplies
        mul!(y, A', x); @test y ≈ M' \ x       # adjoint = MatrixInverse(M')
    end
    D = _diag(n)
    @test MI(MI(D)) === D                       # double inverse collapses to parent
end

@testset "MatrixFactorization" begin
    n = 4
    L = _lower(n); D = _diag(n)
    A = MF(L, D)                                 # represents L*D
    M = Matrix(L) * Matrix(D)
    x = randn(n); y = similar(x)
    @test size(A) == (n, n)
    mul!(y, A, x);  @test y ≈ M * x
    ldiv!(y, A, x); @test y ≈ M \ x
    mul!(y, A', x); @test y ≈ M' * x             # (L*D)' = D'*L'
    # matrix-argument * and \
    X = randn(n, 3)
    @test A * X ≈ M * X
    @test A \ X ≈ M \ X
    # 2-arg in-place ldiv!
    xc = copy(x); ldiv!(A, xc); @test xc ≈ M \ x
    # MatrixInverse of a factorization flips + inverts the factors
    Ai = MI(A)
    mul!(y, Ai, x); @test y ≈ M \ x
end

@testset "SuccessiveReflections" begin
    n = 5
    x = randn(n); y = similar(x)
    # empty ⇒ identity
    S0 = SR(n)
    @test size(S0) == (n, n)
    mul!(y, S0, x);  @test y ≈ x
    ldiv!(y, S0, x); @test y ≈ x
    # one Householder reflection H = I - 2vvᵀ (orthogonal + symmetric ⇒ H⁻¹ = H)
    v = normalize(randn(n))
    S1 = _reflection(v)
    H = I - 2 * v * v'
    mul!(y, S1, x);  @test y ≈ H * x
    ldiv!(y, S1, x); @test y ≈ H \ x
    mul!(y, S1', x); @test y ≈ H' * x            # adjoint = MatrixInverse(S1)
end

@testset "ScaleThenReflect alias" begin
    n = 4
    v = normalize(randn(n))
    S = _reflection(v); D = _diag(n)
    A = MF(S, D)                                  # scale (D) then reflect (S): H*D
    @test A isa STR
    H = I - 2 * v * v'
    M = H * Matrix(D)
    x = randn(n); y = similar(x)
    mul!(y, A, x); @test y ≈ M * x
end

@testset "structural hash/isequal/== (regression for 3b3c05d)" begin
    n = 4
    L = _lower(n); D = _diag(n)
    v = normalize(randn(n)); idxs = [collect(1:n)]
    # Builders producing two structurally-equal-but-distinct operators.
    mk_mf() = MF(copy(L), copy(D))
    mk_mi() = MI(copy(D))
    mk_sr() = SR(deepcopy(idxs), [copy(v)], zeros(n), zeros(n), zeros(n))

    @testset "type $(nameof(typeof(mk())))" for mk in (mk_mf, mk_mi, mk_sr)
        a, b = mk(), mk()
        # Base's AbstractArray hash would index via getindex ⇒ CanonicalIndexError.
        @test hash(a) isa UInt
        @test hash(a) == hash(b)                 # structural, all-fields
        @test isequal(a, b)
        @test a == b
        @test isequal(a, a)
        # THE crash path: operator as a Dict key (DynamicObjects memoize! cache).
        d = Dict(a => 1)
        @test haskey(d, b)
        @test d[b] == 1                          # lookup via structural hash + isequal
    end

    # Realistic nested key from the warmup pipeline:
    # MatrixFactorization(SuccessiveReflections, Diagonal) — the `adaptive` scale option.
    nested() = MF(SR(deepcopy(idxs), [copy(v)], zeros(n), zeros(n), zeros(n)), copy(D))
    dn = Dict(nested() => :ok)
    @test dn[nested()] == :ok

    # Kinetic-energy shape: MatrixFactorization(L, L').
    kd = Dict(MF(copy(L), copy(L)') => 7)
    @test kd[MF(copy(L), copy(L)')] == 7

    # Different concrete expression types compare unequal WITHOUT indexing.
    @test !isequal(mk_mi(), mk_mf())
    @test mk_mi() != mk_mf()

    # The matrix-free contract itself: element indexing is deliberately unsupported.
    @test_throws Base.CanonicalIndexError mk_mf()[1, 1]
    @test_throws Base.CanonicalIndexError mk_sr()[1, 1]
end

# Snag `tsvd-fallback-er-38ce8e53`: when one coordinate is left in the
# reflection loop, `grad_cov_ev` gets a 1 × n block. `tsvd` breaks down on many
# of those, and every breakdown logged an error with a stack trace before the
# power-iteration fallback returned `[1]`. A one-row block now gets `[1]` directly.
@testset "a one-row block gets its singular vector without tsvd's error" begin
    rng = Xoshiro(1)
    blocks = [randn(rng, 1, 1000) .* exp(randn(rng)) .+ randn(rng) for _ in 1:100]
    # Positive control: these blocks include ones `tsvd` itself throws on.
    tsvd_throws(g) = try
        WarmupHMC.tsvd(g; initvec=ones(1))
        false
    catch
        true
    end
    @test count(tsvd_throws, blocks) > 0
    for g in blocks
        v = @test_logs WarmupHMC.grad_cov_ev(nothing, g)
        @test v == [1.0]
        @test v == WarmupHMC._top_left_singular_vector(g)
    end

    # The call site: one badly scaled coordinate among three. The reflection loop
    # reaches the 1 × n block, adds no reflection, and logs nothing.
    n_throwing = 0
    for _ in 1:50
        p = randn(rng, 3, 1000) .* [10.0, 1.0, 1.0]
        g = randn(rng, 3, 1000)
        n_throwing += tsvd_throws(g[1:1, :])
        t = @test_logs WarmupHMC.update_loss!(SR(3), p, g)
        @test isempty(t.reflections)
        @test t.transformation_losses[1] > log(2) > maximum(t.transformation_losses[2:3])
    end
    @test n_throwing > 0
end

# Strongly correlated pairs with scales over two decades, as a window `update_loss!`
# sees: positions `p` and their gradients `g = -Σ⁻¹ p`. Every pair is badly
# fitted on the diagonal, so the reflection fit wants one reflection per pair.
_correlated_pairs_window(rng, d; n=1000, ρ=0.95) = begin
    s = exp.(2 .* rand(rng, d) .- 1)
    z1, z2 = randn(rng, d ÷ 2, n), randn(rng, d ÷ 2, n)
    p = zeros(d, n); g = zeros(d, n)
    for (k, i) in enumerate(1:2:d-1)
        u, v = z1[k, :], ρ .* z1[k, :] .+ sqrt(1 - ρ^2) .* z2[k, :]
        p[i, :], p[i+1, :] = s[i] .* u, s[i+1] .* v
        # precision of the pair's standardized coordinates: [1 -ρ; -ρ 1] / (1 - ρ²)
        g[i, :] = -(u .- ρ .* v) ./ ((1 - ρ^2) * s[i])
        g[i+1, :] = -(v .- ρ .* u) ./ ((1 - ρ^2) * s[i+1])
    end
    p, g
end

@testset "the product applies reflections without copying per reflection" begin
    # The former product, verbatim: it gathered `y[idx]` as a copy for every
    # reflection and the in-place update unaliased a second one (1.29 MB per
    # product at d = 400 with 300 reflections). The shipped product must agree
    # with it bit for bit: the dot is the same contiguous BLAS `dot`.
    reference!(y, A, x, order) = begin
        (; idxs, reflections) = A
        y .= x
        for i in order
            vy = dot(reflections[i], y[idxs[i]])
            y[idxs[i]] .-= 2 .* reflections[i] .* vy
        end
        y
    end
    rng = Xoshiro(3)
    d = 60
    p, g = _correlated_pairs_window(rng, d)
    t = WarmupHMC.update_loss!(SR(d), p, g; max_reflections=typemax(Int))
    @test length(t.reflections) > 10
    for _ in 1:20
        x = randn(rng, d)
        @test mul!(similar(x), t, x) == reference!(similar(x), t, x, reverse(eachindex(t.idxs)))
        @test ldiv!(similar(x), t, x) == reference!(similar(x), t, x, eachindex(t.idxs))
    end
    x = randn(rng, d); y = similar(x)
    z = copy(x)
    @test mul!(z, t, z) == mul!(y, t, x)        # aliased, as `MatrixFactorization` calls it
    mul!(y, t, x)
    @test @allocated(mul!(y, t, x)) <= 8d + 128  # one O(d) buffer, not O(Σ|idx|)
end

@testset "max_reflections bounds the reflection fit" begin
    rng = Xoshiro(4)
    d = 60
    p, g = _correlated_pairs_window(rng, d)
    fit(; kwargs...) = WarmupHMC.update_loss!(SR(d), copy(p), copy(g); kwargs...)
    unbounded = fit(max_reflections=typemax(Int))
    # Positive control: unbounded, this window takes more than the default bound.
    @test length(unbounded.reflections) > WarmupHMC._DEFAULT_MAX_REFLECTIONS
    @test length(fit().reflections) == WarmupHMC._DEFAULT_MAX_REFLECTIONS
    bounded = fit(max_reflections=3)
    @test length(bounded.reflections) == 3
    # The bound stops the greedy loop early: its reflections are the unbounded
    # fit's first ones.
    @test bounded.reflections == unbounded.reflections[1:3]
    @test bounded.idxs == unbounded.idxs[1:3]
    @test isempty(fit(max_reflections=0).reflections)
end
end
