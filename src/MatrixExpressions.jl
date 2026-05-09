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

grad_cov_ev(p, g) = try
    tsvd(g; initvec=ones(size(g, 1)))[1][:, 1]
catch e
    @error "tsvd(...) failed, falling back to eigen(cov(...))"
    eigen(Symmetric(cov(g')), size(g,1):size(g,1)).vectors[:, 1]
    # rethrow()
end
exhaustive_ev(p, g) = begin 
    P = Symmetric(cov(p'))##+1e-8I
    G = Symmetric(cov(g'))#+1e-8I
    eP = eigen(P)
    eG = eigen(G)
    eGPG = eigen(Symmetric(cholesky(G).L * P * cholesky(G).L'))
    ePGP = eigen(Symmetric(cholesky(P).L * G * cholesky(P).L'))
    # ePG = eigen(P,G)
    # eGP = eigen(G,P)
    eAC2 = eigen(Symmetric(P*G+G*P))
    eC = eigen(Symmetric(P/G+G\P))
    eAC = eigen(Symmetric(P\G+G/P))
    # vs = map(normalize!, (
    #     eP.vectors[:, end],
    #     eG.vectors[:, end],
    #     eAC2.vectors[:, end],
    #     eP.vectors[:, 1],
    #     eG.vectors[:, 1],
    #     eAC2.vectors[:, 1],
    #     # ePG.vectors[:, 1],
    #     # ePG.vectors[:, end],
    #     # eGP.vectors[:, 1],
    #     # eGP.vectors[:, end],
    #     eC.vectors[:, end],
    #     eC.vectors[:, 1],
    #     eAC.vectors[:, end],
    #     eAC.vectors[:, 1],
    # ))
    eigens = (eP, eG, eGPG, ePGP, eAC2, eC, eAC)
    vs = map(normalize, eachcol(mapreduce(e->e.vectors, hcat, eigens)))
    loss(v) = abs2(.5*log(v' * P * v * v' * G * v))
    # losses = map(loss, vs)
    # display(eAC2.values')
    display(reshape(round.(map(loss, vs); sigdigits=2), (:, length(eigens))))
    argmin(loss, vs)
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
update_loss!(t::Diagonal, p, g; kwargs...) = mean(1:LinearAlgebra.checksquare(t)) do i 
    pi, gi = view(p, i, :), view(g, i, :)
    s1, s2 = std(pi), std(gi)
    if s2 == 0
        # @warn "Assuming Laplace" i s1 s2 mean(gi)
        t[i,i] = sqrt(2.) / abs(mean(gi))
        return Inf
    end
    # s1 = std(pi; mean=cv_mean(pi, gi))
    # @info (i, cor(pi, gi), mean(pi)=>cv_mean(pi, gi), std(pi)=>s1)
    t[i,i] = sqrt(s1 / s2)
    (s1 * s2)
end
update_loss!(t::MatrixFactorization, p, g; kwargs...) = update_loss!(t.m2, t.m1 \ p, t.m1' * g; kwargs...)
update_loss!(t::ScaleThenReflect, p, g; kwargs...) = begin
    update_loss!(t.m1, p, g; kwargs...)
    return update_loss!(t.m2, p, g; kwargs...)
end
