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
