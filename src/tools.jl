struct ConjugateLinearRegression{T<:Real}
    potential::Vector{T}
    precision::Matrix{T}
    ab::Vector{T}
    location::Vector{T}
    L::LowerTriangular{T, Matrix{T}}
end
ConjugateLinearRegression(n; a=1e-3, b=1e-3) = ConjugateLinearRegression(zeros(n), zeros((n,n)), [a, b, 0.], zeros(n), LowerTriangular(zeros((n,n))))
ConjugateLinearRegression(;
    precision::Matrix, 
    potential::Vector=zeros(LinearAlgebra.checksquare(precision)), 
    a=1e-3, 
    b=1e-3
) = ConjugateLinearRegression(potential, precision, [a, b, 0.], 0*potential, LowerTriangular(0*precision))
condition!(p::ConjugateLinearRegression, X::AbstractMatrix, y::AbstractVector) = begin
    (n, o) = size(X)
    @assert o == length(p.location)
    @assert n == length(y)
    p.potential .+= X' * y
    p.precision .+= X' * X
    p.ab[1] += n/2
    p.ab[2] += .5 * sum(abs2, y)
    p.ab[3] = 0.
    p
end
condition!(p::ConjugateLinearRegression, X::AbstractVector, y::AbstractVector) = condition!(p, X; n=length(y), sum=sum(y), sum_abs2=sum(abs2, y))
condition!(p::ConjugateLinearRegression, X::AbstractVector; n, sum, sum_abs2) = begin 
    @assert length(X) == length(p.potential)
    @assert isfinite(sum) (;n, sum, sum_abs2)
    @assert isfinite(sum_abs2) (;n, sum, sum_abs2)
    p.potential .+= X .* sum
    p.precision .+= n .* X .* X'
    p.ab[1] += n/2
    p.ab[2] += .5 * sum_abs2
    p.ab[3] = 0.
    p
end
prepare!(p::ConjugateLinearRegression) = if p.ab[3] == 0
    parent(p.L) .= Symmetric(p.precision)
    @assert isposdef!(parent(p.L)') "Precision matrix is not positive definite!"
    # parent(p.L) .= cholesky(Symmetric(p.precision)).L
    ldiv!(p.location, p.L, p.potential)
    p.ab[3] = p.ab[2] - .5 * sum(abs2, p.location)
    ldiv!(p.L', p.location)
    p
else
    p
end
location!(p::ConjugateLinearRegression) = prepare!(p).location
location!(p::ConjugateLinearRegression, x::AbstractVector) = dot(x, location!(p))
inv_scale!(p::ConjugateLinearRegression) = prepare!(p).L'
obsvardist!(p::ConjugateLinearRegression) = begin 
    prepare!(p)
    InverseGamma(p.ab[1], p.ab[3])
end
obsscale!(p::ConjugateLinearRegression) = sqrt(mode(obsvardist!(p)))
locscale!(p::ConjugateLinearRegression, x::AbstractVector; y=zero(x)) = dot(x, location!(p)), obsscale!(p) * norm(ldiv!(y, inv_scale!(p)', x))
mlocation!(p::ConjugateLinearRegression, x::AbstractVector; kwargs...) = Normal(
    locscale!(p, x; kwargs...)...
)
qlocation!(p::ConjugateLinearRegression, x::AbstractVector, q; kwargs...) = quantile(
    mlocation!(p::ConjugateLinearRegression, x::AbstractVector; kwargs...), q
)
plocscale!(p::ConjugateLinearRegression, x::AbstractVector; kwargs...) = begin
    loc, scale = locscale!(p, x; kwargs...)
    loc, sqrt(scale^2 + obsscale!(p)^2)
end
mpred!(p::ConjugateLinearRegression, x::AbstractVector; kwargs...) = Normal(
    plocscale!(p, x; kwargs...)...
)
qpred!(p::ConjugateLinearRegression, x::AbstractVector, q; kwargs...) = quantile(
    mpred!(p::ConjugateLinearRegression, x::AbstractVector; kwargs...), q
)

abstract type AbstractGPKernel{T} end
AbstractScalarGPKernel{T<:Real} = AbstractGPKernel{T}
x_scale(kernel::AbstractGPKernel) = kernel.x_scale
y_scale(kernel::AbstractGPKernel) = kernel.y_scale
cov_kernel(kernel::AbstractGPKernel) = kernel
y_kernel(kernel::AbstractGPKernel) = kernel

struct SquaredExponentialKernel{T} <: AbstractGPKernel{T}
    x_scale::T
    y_scale::T
end
(k::SquaredExponentialKernel)(x::Real, y::Real) = k.y_scale^2 * exp(-.5 * abs2((x - y) / k.x_scale))

struct IntegratedSquaredExponentialKernel{N,T} <: AbstractGPKernel{T}
    x_scale::T
    y_scale::T
    IntegratedSquaredExponentialKernel(N, x_scale, y_scale) = new{N,typeof(x_scale)}(x_scale, y_scale)
end
cov_kernel(k::IntegratedSquaredExponentialKernel) = SquaredExponentialKernel(k.x_scale, k.y_scale)
y_kernel(kernel::IntegratedSquaredExponentialKernel{0}) = cov_kernel(kernel)
using Distributions
# Phi(x) = .5 * (1 + erf(x/sqrt(2)))
# F(x) = Phi((x - loc) / scale) = .5 * (1 + erf((x - loc)/(scale*sqrt(2))))
(k::IntegratedSquaredExponentialKernel{1})(x::Real, y::Real) = k.y_scale^2*(sqrt(2pi)*k.x_scale)*(cdf(Normal(x, k.x_scale), y)) 
# G(x) = .5 * (x + ERF((x - loc)/(scale*sqrt(2)))*(scale*sqrt(2)))
# ERF(x) = x * erf(x) + exp(-x^2)/sqrt(pi) + C
ERF(x) = x * Distributions.erf(x) + exp(-x^2)/sqrt(pi)
(k::IntegratedSquaredExponentialKernel{2})(x::Real, y::Real) = k.y_scale^2*(sqrt(2pi)*k.x_scale)*(
    .5 * ((y-x) + ERF((y - x)/(k.x_scale*sqrt(2)))*(k.x_scale*sqrt(2)))
)
struct TransformedGPKernel{F,T,K<:AbstractGPKernel{T}} <: AbstractGPKernel{T}
    func::F
    kernel::K
end
x_scale(kernel::TransformedGPKernel) = x_scale(kernel.kernel)
y_scale(kernel::TransformedGPKernel) = y_scale(kernel.kernel)
cov_kernel(k::TransformedGPKernel) = cov_kernel(k.kernel)
(k::TransformedGPKernel)(x::Real, y::Real) = k.func(y, y_kernel(k.kernel)(x, y))


"""
Inducing point GP regression
"""
struct IPGPRegression{L,F<:Tuple,K<:AbstractGPKernel,T<:Real,C}
    link::L
    functions::F
    kernel::K
    inducing_x::Vector{T}
    inducing_L::LowerTriangular{T, Matrix{T}}
    clr::ConjugateLinearRegression{T}
    cache::C
end
identity2(x, y) = y
link(gp::IPGPRegression, x) = Base.Fix1(gp.link, x)
inducing_x(lo, hi, x_scale; n_pad=1) = range(
    ((lo, hi) .+ (-x_scale, +x_scale) .* n_pad)..., n_pad[1] + n_pad[end] + 1 + ceil(Int, (hi-lo)/x_scale)
)
IPGPRegression(lo, hi; link=identity2, functions=(), kernel=SquaredExponentialKernel(1., 1.), n_pad=1) = IPGPRegression(
    link,
    functions,
    kernel,
    inducing_x(lo, hi, x_scale(kernel); n_pad)
    # range(lo-n_pad*x_scale(kernel), hi+n_pad*x_scale(kernel), 2n_pad + 1 + ceil(Int, (hi-lo)/x_scale(kernel)))
)
IPGPRegression(link, functions, kernel, inducing_x; nugget=1e-8, T=eltype(inducing_x)) = IPGPRegression(
    link, functions, kernel, collect(inducing_x),
    cholesky(cov_kernel(kernel).(inducing_x, inducing_x') + y_scale(kernel)^2 * nugget * I).L,
    ConjugateLinearRegression(;precision=Matrix(Diagonal(ones(length(functions)+length(inducing_x))))),
    (;n=Int[], x=T[], sum=T[], sum_abs2=T[], scale=T[], X=zeros(length(functions)+length(inducing_x)))
)
Base.merge(a::IPGPRegression, rgs::IPGPRegression...) = begin
    gp = IPGPRegression(a.link, a.functions, a.kernel, Float64[])
    args = (a, rgs...)
    map(append!, gp.cache, (arg.cache for arg in args)...)
    prepare!!(gp, inducing_x(minimum(x->x.inducing_x[1], args), maximum(x->x.inducing_x[end], args), x_scale(gp.kernel)))
end
X!(gp::IPGPRegression, x::Real) = begin 
    (;functions, kernel, inducing_x, cache, inducing_L) = gp
    (;X) = cache
    for (i, f) in enumerate(functions)
        X[i] = f(x)
    end
    @views gp_X = X[length(functions)+1:end]
    gp_X .= y_kernel(kernel).(inducing_x, x)
    ldiv!(inducing_L, gp_X)
    X
end
condition!(gp::IPGPRegression, x::Real, y; kwargs...) = condition!(
    gp, x; 
    n=length(y), sum=sum(link(gp, x), y), sum_abs2=sum(abs2 ∘ link(gp, x), y), kwargs...
)
condition!(gp::IPGPRegression, x::Real; n, sum, sum_abs2, new=true) = begin
    (;functions, kernel, clr, cache) = gp
    condition!(clr, X!(gp, x); n, sum, sum_abs2)
    new && map(
        push!, 
        (;cache.x, cache.n, cache.sum, cache.sum_abs2, cache.scale), 
        (;x, n, sum, sum_abs2, scale=norm(cache.X[1+length(functions):end]) / y_scale(kernel))
    )
    gp
end
condition!(gp::IPGPRegression, x::AbstractArray, y::AbstractArray) = for (xi, yi) in zip(x, y)
    condition!(gp, xi, yi)
end
location!(gp::IPGPRegression, x::Real) = location!(gp.clr, X!(gp, x))
locscale!(gp::IPGPRegression, x::Real) = locscale!(gp.clr, X!(gp, x); y=gp.cache.X)
mlocation!(gp::IPGPRegression, x::Real) = mlocation!(gp.clr, X!(gp, x); y=gp.cache.X)
qlocation!(gp::IPGPRegression, x::Real, q::Real) = qlocation!(gp.clr, X!(gp, x), q; y=gp.cache.X)
mcdf!(gp::IPGPRegression, x::Real, y::Real) = cdf(
    mlocation!(gp.clr, X!(gp, x); y=gp.cache.X), 
    gp.link(x, y)
)
mpred!(gp::IPGPRegression, x::Real) = mpred!(gp.clr, X!(gp, x); y=gp.cache.X)
qpred!(gp::IPGPRegression, x::Real, q::Real) = qpred!(gp.clr, X!(gp, x), q; y=gp.cache.X)
pcdf!(gp::IPGPRegression, x::Real, y::Real) = cdf(
    mpred!(gp, x), 
    gp.link(x, y)
)
prepare!!(
    gp::IPGPRegression, x::Real; functions=gp.functions
) = prepare!!(
    gp, 
    gp.inducing_x[1] < x < gp.inducing_x[end] ? gp.inducing_x : inducing_x(
        extrema((gp.inducing_x[1], x, gp.inducing_x[end]))..., x_scale(gp.kernel); n_pad=0
    ); 
    functions
)
prepare!!(
    gp::IPGPRegression, new_inducing_x::AbstractVector; functions=gp.functions
) = if functions != functions || gp.inducing_x != new_inducing_x
    (;link, kernel, cache) = gp
    gp = IPGPRegression(
        link, functions, kernel, new_inducing_x,
        # extrema((inducing_x[2], x, inducing_x[end-1]))...; 
        # link, functions, kernel, 
    )
    # @info "Refitting GP ($(length(inducing_x))=>$(length(gp.inducing_x)))..."
    for (i, (x, n, sum, sum_abs2)) in enumerate(
        zip(cache.x, cache.n, cache.sum, cache.sum_abs2)
    )
        condition!(gp, x; n, sum, sum_abs2)
        cache.scale[i] = norm(cache.X[1+length(functions):end]) / y_scale(kernel)
    end
    gp
else
    gp
end
rescale!!(kernel::SquaredExponentialKernel, new_y_scale::Real) = SquaredExponentialKernel(x_scale(kernel), new_y_scale)
rescale!!(kernel::IntegratedSquaredExponentialKernel{N}, new_y_scale::Real) where {N} = IntegratedSquaredExponentialKernel(N, x_scale(kernel), new_y_scale)
rescale!!(kernel::TransformedGPKernel, new_y_scale::Real) = TransformedGPKernel(kernel.func, rescale!!(kernel.kernel, new_y_scale))
rescale!!(gp::IPGPRegression; kwargs...) = rescale!!(gp, autoscale(gp, ;kwargs...))
rescale!!(gp::IPGPRegression, new_y_scale::Real) = if y_scale(gp.kernel) == new_y_scale
    gp
else
    @assert new_y_scale > 0
    (;link, functions, kernel, inducing_x, inducing_L, clr, cache) = gp
    # @assert length(functions) == 0
    (;potential, precision, ab) = clr
    f_idxs = 1:length(functions)
    gp_idxs = 1+length(functions):length(potential)
    ratio = new_y_scale / y_scale(gp.kernel)
    kernel = rescale!!(kernel, new_y_scale)
    parent(inducing_L) .*= ratio
    potential[gp_idxs] .*= ratio
    # precision .= I + (precision - I) .* ratio^2 = I * (1 - ratio^2) .+ precision .* ratio^2
    precision[f_idxs, gp_idxs] .*= ratio
    precision[gp_idxs, f_idxs] .*= ratio
    precision[gp_idxs, gp_idxs] .*= ratio^2
    precision[diagind(precision)[gp_idxs]] .+= 1 - ratio^2
    ab[3] = 0.
    IPGPRegression(
        link, functions, kernel, inducing_x, inducing_L, clr, cache
    )
end
autoscale((;functions, kernel, inducing_x, inducing_L, clr, cache)::IPGPRegression; R2=.99, n=1) = if n == 0
    y_scale(kernel)
else 
    beta = location!(clr)
    # return clamp(
    #     y_scale(kernel) * norm(beta[1+length(functions):end]) * sqrt(length(inducing_L)),
    #     .5 * y_scale(kernel),
    #     2 * y_scale(kernel)
    # )
    # @assert isa(kernel, SquaredExponentialKernel)
    n_sum, sum_sum, sum_abs2_sum = 0, 0., 0.
    (;X) = cache

    for (x, n, sum, sum_abs2, scale) in zip(cache.x, cache.n, cache.sum, cache.sum_abs2, cache.scale)
        y = 0.
        for (w, f) in zip(beta, functions)
            y += w * f(x)
        end
        n_sum += n
        # sum_sum += sum - n*y
        sum_abs2_sum += (sum_abs2 -2sum*y + n*abs2(y)) / (scale)^2
    end
    y_var = sum_abs2_sum / n_sum# - (sum_sum / n_sum)^2
    obs_var = obsscale!(clr)^2
    rv = if y_var > obs_var > 0# && isa(kernel, SquaredExponentialKernel)
        # @info kernel.y_scale=>sqrt((y_var - .5obs_var)/obs_var)
        # @info isa(kernel, SquaredExponentialKernel)=>(y_var - obs_var)/y_var
        sqrt((y_var - obs_var)/obs_var)
        # sqrt(R2*y_var/obs_var)
        # sqrt(R2*y_var/obs_var)
    elseif y_var > 0
        # @info "$y_var > $obs_var ($(isa(kernel, SquaredExponentialKernel)))"
        sqrt(R2*y_var/obs_var)
    else
        y_scale(kernel)
        # @info y_var
        # y_scale(kernel)
        # sqrt(y_var/2)
    end
    # @info (y_var - obs_var)/y_var
    # @info (kernel) => rv
    clamp(rv, .9^n*y_scale(kernel), 1.1^n*y_scale(kernel))
end





broadcastable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
broadcastable(x::Symbol) = (!Base.isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
broadcastable(x::Expr) = x.head !== :$
unbroadcast(x) = x
function unbroadcast(x::Expr)
    if x.head === :.=
        Expr(:(=), x.args...)
    elseif x.head === :block # occurs in for x=..., y=...
        Expr(:block, Base.mapany(unbroadcast, x.args)...)
    else
        x
    end
end
__broadcasted__(x) = x
function __broadcasted__(x::Expr)
    broadcasted = :(Base.broadcasted)
    broadcastargs = Base.mapany(__broadcasted__, x.args)
    return if x.head === :call && broadcastable(x.args[1])
        Expr(:call, broadcasted, broadcastargs...)
    elseif x.head === :comparison
        error()
        Expr(:comparison, (iseven(i) && broadcastable(arg) && arg isa Symbol && Base.isoperator(arg) ?
                               Symbol('.', arg) : arg for (i, arg) in pairs(broadcastargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], broadcastargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        Expr(:call, broadcasted, x.head, broadcastargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(:call, broadcasted, x.head, broadcastargs...)
        else
            Expr(x.head, broadcastargs...)
        end
    end
end
macro broadcasted(x)
    esc(__broadcasted__(x))
end
macro bsum(x)
    :(sum($(esc(__broadcasted__(x)))))
end