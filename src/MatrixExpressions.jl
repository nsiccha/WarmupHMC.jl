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