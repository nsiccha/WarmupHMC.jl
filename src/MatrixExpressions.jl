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
LinearAlgebra.mul!(y::AbstractVector, A::SuccessiveReflections, x::AbstractVector) = begin
    (;idxs, reflections) = A
    y .= x
    for i in reverse(eachindex(idxs))
        vy = dot(reflections[i], y[idxs[i]])
        y[idxs[i]] .-= 2 .* reflections[i] .* vy
    end
    y
end
LinearAlgebra.ldiv!(y::AbstractVector, A::SuccessiveReflections, x::AbstractVector) = begin
    (;idxs, reflections) = A
    y .= x
    for i in (eachindex(idxs))
        vy = dot(reflections[i], y[idxs[i]])
        y[idxs[i]] .-= 2 .* reflections[i] .* vy
    end
    y
end
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
grad_cov_ev(p, g) = try
    tsvd(g; initvec=ones(size(g, 1)))[1][:, 1]
catch e
    @error "tsvd(...) failed, falling back to power iteration" size(g) exception=(e, catch_backtrace())
    _top_left_singular_vector(g)
end
update_loss!(t::SuccessiveReflections, p, g; threshold=log(2), v_f=grad_cov_ev, idx_f=v->argmax(v.^2), kwargs...) = begin 
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
