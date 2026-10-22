abstract type AbstractMatrixExpression{T} <: AbstractMatrix{T} end
Base.show(io::IO, ::MIME"text/plain", A::AbstractMatrixExpression) = show(io, A)
Base.size(A::AbstractMatrixExpression, i) = size(A)[i]
struct MatrixInverse{T,M<:AbstractMatrix{T}} <: AbstractMatrixExpression{T}
    parent::M
end
struct MatrixFactorization{T,M1<:AbstractMatrix{T},M2<:AbstractMatrix{T}} <: AbstractMatrixExpression{T}
    m1::M1
    m2::M2
end
struct SuccessiveReflections{T,I} <: AbstractMatrixExpression{T}
    idxs::Vector{Vector{I}}
    reflections::Vector{Vector{T}}
    s1::Vector{T}
    s2::Vector{T}
    transformation_losses::Vector{T}
end

# These are matrix-free operators: they subtype `AbstractMatrix` for the
# operator interface (`mul!`/`ldiv!`) but deliberately define no `getindex`.
# Base's generic `AbstractArray` `hash`/`isequal`/`==` iterate the elements via
# `getindex` and therefore throw, which crashes as soon as one lands in a Dict
# key (e.g. DynamicObjects' memoization cache). Define structural (all-fields)
# hashing and equality instead — well-founded because every field is either a
# plain array or another `AbstractMatrixExpression`.
Base.hash(A::AbstractMatrixExpression, h::UInt) = begin
    h = hash(typeof(A), h)
    for i in 1:nfields(A)
        h = hash(getfield(A, i), h)
    end
    h
end
Base.isequal(A::T, B::T) where {T<:AbstractMatrixExpression} =
    all(i -> isequal(getfield(A, i), getfield(B, i)), 1:nfields(A))
Base.:(==)(A::T, B::T) where {T<:AbstractMatrixExpression} =
    all(i -> getfield(A, i) == getfield(B, i), 1:nfields(A))
# Different concrete expression types are structurally distinct (consistent
# with hashing `typeof` above); decide without touching `getindex`.
Base.isequal(::AbstractMatrixExpression, ::AbstractMatrixExpression) = false
Base.:(==)(::AbstractMatrixExpression, ::AbstractMatrixExpression) = false

Base.show(io::IO, A::MatrixInverse) = print(io, "MatrixInverse($(parent(A)))")
Base.size(A::MatrixInverse, args...) = size(parent(A), args...)
Base.parent(A::MatrixInverse) = A.parent
Base.adjoint(A::MatrixInverse) = MatrixInverse(parent(A)')
MatrixInverse(A::MatrixInverse) = parent(A)
LinearAlgebra.mul!(y::AbstractVector, A::MatrixInverse, x::AbstractVector) = ldiv!(y,parent(A),x)
LinearAlgebra.ldiv!(y::AbstractVector, A::MatrixInverse, x::AbstractVector) = mul!(y,parent(A),x)

Base.show(io::IO, A::MatrixFactorization) = print(io, "MatrixFactorization(", A.m1, " * ", A.m2, ").")
Base.size(A::MatrixFactorization) = (size(A.m1, 1), size(A.m2, 2))
Base.adjoint(A::MatrixFactorization) = MatrixFactorization(A.m2', A.m1')
MatrixInverse(A::MatrixFactorization) = MatrixFactorization(MatrixInverse(A.m2), MatrixInverse(A.m1))
LinearAlgebra.mul!(y::AbstractVector, A::MatrixFactorization, x::AbstractVector) = begin 
    mul!(y, A.m2, x)
    mul!(y, A.m1, y)
    y
end
LinearAlgebra.ldiv!(y::AbstractVector, A::MatrixFactorization, x::AbstractVector) = begin 
    ldiv!(y, A.m1, x)
    ldiv!(y, A.m2, y)
    y
end
LinearAlgebra.ldiv!(A::MatrixFactorization, x::AbstractVector) = ldiv!(A.m2, ldiv!(A.m1, x))
Base.:\(A::MatrixFactorization, x::AbstractMatrix) = begin 
    y = zero(x)
    for (yi, xi) in zip(eachcol(y), eachcol(x))
        ldiv!(yi, A, xi)
    end
    y
end
Base.:*(A::MatrixFactorization, x::AbstractMatrix) = begin 
    y = zero(x)
    for (yi, xi) in zip(eachcol(y), eachcol(x))
        mul!(yi, A, xi)
    end
    y
end

SuccessiveReflections(n::Int64) = SuccessiveReflections(
    Vector{Vector{Int64}}(),
    Vector{Vector{Float64}}(),
    Vector{Float64}(undef,n),
    Vector{Float64}(undef,n),
    Vector{Float64}(undef,n)
)
Base.show(io::IO, A::SuccessiveReflections) = print(io, "SuccessiveReflections with $(length(A.idxs)) reflections.")
Base.size(A::SuccessiveReflections) = (length(A.s1), length(A.s1))
Base.adjoint(A::SuccessiveReflections) = MatrixInverse(A)
# Apply reflections `order` (indices into `A.idxs`) to `x`, writing `y`. Each
# reflection is y[idx] -= 2 r (r ⋅ y[idx]). The gathered y[idx] goes into ONE
# buffer, sized by the first (largest: the bad set only shrinks) index set, so
# the product allocates O(d) once instead of two copies per reflection, and the
# dot is still a contiguous BLAS `dot` on the same values: bit-identical to the
# former `dot(r, y[idx])` / `y[idx] .-= 2 .* r .* vy`.
_apply_reflections!(y, A::SuccessiveReflections, x, order) = begin
    (;idxs, reflections) = A
    y .= x
    isempty(idxs) && return y
    buffer = similar(y, length(first(idxs)))
    for i in order
        idx, r = idxs[i], reflections[i]
        gathered = view(buffer, 1:length(idx))
        @inbounds for (k, j) in enumerate(idx)
            gathered[k] = y[j]
        end
        vy = dot(r, gathered)
        @inbounds for (k, j) in enumerate(idx)
            y[j] -= 2 * r[k] * vy
        end
    end
    y
end
LinearAlgebra.mul!(y::AbstractVector, A::SuccessiveReflections, x::AbstractVector) =
    _apply_reflections!(y, A, x, reverse(eachindex(A.idxs)))
LinearAlgebra.ldiv!(y::AbstractVector, A::SuccessiveReflections, x::AbstractVector) =
    _apply_reflections!(y, A, x, eachindex(A.idxs))
ScaleThenReflect{T,I,V} = MatrixFactorization{T,SuccessiveReflections{T,I},Diagonal{T,V}}

# Top left singular vector of `g` (d × n): the direction `tsvd(g)[1][:, 1]`
# returns. Power iteration on `g * g'`, applied as two products, never forms a
# d × d matrix: O(d n) per iteration (linear-in-d rule, decision `1o12joq`).
function _top_left_singular_vector(g; maxiter=1000, tol=1e-12)
    v = normalize!(ones(size(g, 1)))
    w, u = similar(v), zeros(eltype(v), size(g, 2))
    for _ in 1:maxiter
        mul!(u, g', v)
        mul!(w, g, u)
        nw = norm(w)
        nw > 0 || return v
        w ./= nw
        converged = 1 - abs(dot(w, v)) < tol
        v, w = w, v
        converged && break
    end
    v
end

# The fallback used to be `eigen(Symmetric(cov(g')))`: a dense d × d covariance
# and eigendecomposition (O(d²) memory, O(d³) time), and the top eigenvector of
# the CENTERED covariance, a different vector from the uncentered singular
# vector `tsvd` returns (todo `0hsdtfg`). The power iteration computes the same
# quantity as `tsvd`, linearly.
#
# A 1 × n block (one coordinate left in `update_loss!`'s reflection loop) has a
# one-dimensional column space, so its top left singular vector is `[1]` — the
# vector the fallback returns for it, exactly. `tsvd` cannot compute it: the first
# Lanczos step already spans that space, so the next residual is zero or rounding
# noise, normalizing it puts NaN or garbage into the bidiagonal, and LAPACK's
# `bdsdc!` throws `LAPACKException(2)` on a sizeable share of 1 × n blocks (and
# not on blocks with two or more rows, rank-deficient ones included). Each such
# call logged an error with a stack trace (snag `tsvd-fallback-er-38ce8e53`).
grad_cov_ev(p, g) = size(g, 1) == 1 ? ones(1) : try
    tsvd(g; initvec=ones(size(g, 1)))[1][:, 1]
catch e
    @error "tsvd(...) failed, falling back to power iteration" size(g) exception=(e, catch_backtrace())
    _top_left_singular_vector(g)
end
# How many reflections one fit may add, unless the sampler is told otherwise
# (`max_reflections`). Every reflection is stored over the coordinates that were
# still badly fitted when it was found, and every metric product applies all of
# them, so storage, a product and the fit itself are O(d · max_reflections): linear
# in the dimension for a fixed bound (decision `1o12joq`). Unbounded, a strongly
# correlated target keeps adding reflections until ~0.47·d² entries are stored at
# d = 400 (go/no-go `04q64y5`). The user asked for a bound that is smallish by
# default and tunable.
const _DEFAULT_MAX_REFLECTIONS = 16
update_loss!(t::SuccessiveReflections, p, g; threshold=log(2), v_f=grad_cov_ev, idx_f=v->argmax(v.^2), max_reflections=_DEFAULT_MAX_REFLECTIONS, kwargs...) = begin
    (;idxs, reflections, s1, s2, transformation_losses) = t
    dimension = LinearAlgebra.checksquare(t)
    s1 .= std.(eachrow(p))
    s2 .= std.(eachrow(g))
    @. transformation_losses = abs2(log(s1 * s2))
    bad_idxs = collect(1:dimension)
    # empty!(splits1)
    empty!(idxs)
    empty!(reflections)
    @views while length(bad_idxs) > 0
        filter!(i->transformation_losses[i]>=threshold, bad_idxs)
        length(bad_idxs) == 0 && break
        length(idxs) >= max_reflections && break

        bad_p = p[bad_idxs, :]
        bad_g = g[bad_idxs, :]
        v = v_f(bad_p, bad_g)
        l = abs2(log(std(v' * bad_p) * std(v' * bad_g)))
        # display((;n=length(bad_idxs),threshold,v_f) => l)
        l > threshold && break

        push!(idxs, copy(bad_idxs))
        v[idx_f(v)] -= -sign(v[idx_f(v)])
        normalize!(v)
        push!(reflections, v)
        vp = v' * bad_p
        vg = v' * bad_g
        bad_p .-= 2 .* v * vp 
        bad_g .-= 2 .* v * vg 
        s1[bad_idxs] .= std.(eachrow(bad_p))
        s2[bad_idxs] .= std.(eachrow(bad_g))
        @. transformation_losses[bad_idxs] = abs2(log(s1[bad_idxs] * s2[bad_idxs]))
    end
    t
end
cv_mean(x1, x2) = begin 
    m1 = mean(x1)
    m2 = mean(x2)
    m12 = mean(x1 .* x2)
    m22 = mean(abs2, x2)
    (m1 - m12/m22*m2)/(1-m2^2/m22)
end
# A diagonal scale entry the kinetic energy can use: DynamicHMC divides by it on
# every momentum draw (`rand_p` → `ldiv!`), so a zero is a `SingularException`, a
# value below `floatmin` overflows the momentum, and a non-finite one poisons
# every position the trajectory visits.
_usable_scale(s) = isfinite(s) && s >= floatmin(s)

# Per coordinate, the window's position spread `s1` and gradient spread `s2`
# estimate the scale `sqrt(s1/s2)` and the loss `s1*s2`: for a Gaussian
# coordinate with standard deviation σ, s1 ≈ σ and s2 ≈ 1/σ, so the estimate is σ
# and the loss is 1 when the frame fits. Degenerate evidence — non-finite spreads,
# an estimate that under- or overflows — yields no usable scale: the coordinate
# keeps its previous one. Any degenerate coordinate, including one whose recorded
# positions never moved (s1 = 0), makes the frame score `Inf`, so it can never
# look BETTER fitted than a frame with real evidence (s1 = 0 used to score 0, the
# best loss possible, and won).
update_loss!(t::Diagonal, p, g; kwargs...) = mean(1:LinearAlgebra.checksquare(t)) do i
    pi, gi = view(p, i, :), view(g, i, :)
    s1, s2 = std(pi), std(gi)
    # s1 = std(pi; mean=cv_mean(pi, gi))
    # @info (i, cor(pi, gi), mean(pi)=>cv_mean(pi, gi), std(pi)=>s1)
    scale, loss = if s2 == 0
        # Constant gradient: assume Laplace.
        sqrt(2.) / abs(mean(gi)), Inf
    elseif s1 == 0
        # The positions never moved, so the previous scale was too small to
        # resolve even one step there, and keeping it keeps the coordinate frozen
        # for good. The gradient still varied: size the coordinate from its spread
        # alone, 1/std(gradient) (the scale a Gaussian's gradient spread implies),
        # never below the scale that was already too small. A restart point, not
        # evidence of fit, so the frame still scores `Inf`.
        max(t[i,i], inv(s2)), Inf
    else
        sqrt(s1 / s2), s1 * s2
    end
    _usable_scale(scale) || return Inf
    t[i,i] = scale
    loss
end
update_loss!(t::MatrixFactorization, p, g; kwargs...) = update_loss!(t.m2, t.m1 \ p, t.m1' * g; kwargs...)
update_loss!(t::ScaleThenReflect, p, g; kwargs...) = begin
    update_loss!(t.m1, p, g; kwargs...)
    return update_loss!(t.m2, p, g; kwargs...)
end

# EXPERIMENTAL (investigation todo `2026-10-22T10-55-12-401-1ruyk25`): fit the
# successive reflections in a diagonally scaled frame. The reflection directions
# are the top singular vectors of the window's GRADIENTS, which in the raw frame
# rank coordinates by their marginal scale (a coordinate with standard deviation
# σ has gradients of size 1/σ), whereas the per-coordinate loss that picks the
# bad coordinates is scale-free. Fitting a diagonal first makes the direction
# search scale-free as well.
#
#   ScaledScaleThenReflect   L = D₀ R D: diagonal, then reflections + diagonal in
#                            its frame (one gradient TSVD per reflection, all in
#                            the scaled frame).
#   TwoStageScaleThenReflect L = R₁ D₁ R₂ D₂: the current fit, then a second
#                            reflections + diagonal fit in its frame (a second
#                            TSVD pass after the diagonal scaling step).
#
# Both stay O(d) per stored factor entry and O(d n) per reflection, like
# `ScaleThenReflect`. Stage 1 runs on copies: `update_loss!(::ScaleThenReflect)`
# reflects its inputs in place.
ScaledScaleThenReflect{T,V} = MatrixFactorization{T,Diagonal{T,V},<:ScaleThenReflect}
TwoStageScaleThenReflect{T} = MatrixFactorization{T,<:ScaleThenReflect,<:ScaleThenReflect}
update_loss!(t::Union{ScaledScaleThenReflect,TwoStageScaleThenReflect}, p, g; kwargs...) = begin
    update_loss!(t.m1, copy(p), copy(g); kwargs...)
    return update_loss!(t.m2, t.m1 \ p, t.m1' * g; kwargs...)
end
n_reflections(t::ScaleThenReflect) = length(t.m1.reflections)
n_reflections(t::ScaledScaleThenReflect) = n_reflections(t.m2)
n_reflections(t::TwoStageScaleThenReflect) = n_reflections(t.m1) + n_reflections(t.m2)

_scale_then_reflect(dimension) = MatrixFactorization(SuccessiveReflections(dimension), Diagonal(ones(dimension)))
_scaled_scale_then_reflect(dimension) = MatrixFactorization(Diagonal(ones(dimension)), _scale_then_reflect(dimension))
_two_stage_scale_then_reflect(dimension) = MatrixFactorization(_scale_then_reflect(dimension), _scale_then_reflect(dimension))
# The `:adaptive` linear transformation candidate both samplers fit each window.
_adaptive_scale(dimension) = _scale_then_reflect(dimension)
