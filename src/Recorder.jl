initialize!(x::NamedTuple, args...) = map(xi->initialize!(xi, args...), x)
pre_move!!(x::NamedTuple, args...) = map(xi->pre_move!!(xi, args...), x)
post_move!!(x::NamedTuple, args...) = map(xi->post_move!!(xi, args...), x)
pre_move!!(x, args...) = x

abstract type AbstractRecorder end
abstract type AbstractInitialRecorder <: AbstractRecorder end
abstract type AbstractContinuousRecorder <: AbstractRecorder end
abstract type AbstractFinalizer <: AbstractRecorder end
initialize!(x::AbstractInitialRecorder, z) = if isa(x.value, Ref)
    x.value[] = x.func(z)
else
    x.value .= x.func(z)
end
initialize!(x::AbstractContinuousRecorder, z) = reset!(x.value)
post_move!!(x::AbstractInitialRecorder, z, f) = x
post_move!!(x::AbstractContinuousRecorder, z, f) = f(x.value, x.func(z))

struct InitialRecorder{F,V} <: AbstractInitialRecorder
    func::F
    value::V
    InitialRecorder(func, value) = new{typeof(func), typeof(value)}(func, value)
    InitialRecorder(f; n) = InitialRecorder(f, single_value(f, n))
end
struct ContinuousRecorder{F,V} <: AbstractContinuousRecorder
    func::F
    value::V
    ContinuousRecorder(func, value) = new{typeof(func), typeof(value)}(func, value)
    ContinuousRecorder(f; n) = ContinuousRecorder(f, multiple_values(f, n))
end

position(z::DynamicHMC.PhasePoint) = z.Q.q
gradient(z::DynamicHMC.PhasePoint) = z.Q.∇ℓq
lpdf(z::DynamicHMC.PhasePoint) = z.Q.ℓq

single_value(::typeof(position), n; T=Float64) = zeros(T, n)
single_value(::typeof(gradient), n; T=Float64) = zeros(T, n)
single_value(::typeof(lpdf), n; T=Float64) = Ref(zero(T))
multiple_values(f, n; kwargs...) = multiple_values(single_value(f, n; kwargs...))
multiple_values(x::Ref) = zeros(eltype(x), 0)
multiple_values(x::Vector) = ElasticMatrix(zeros(eltype(x), (length(x), 0)))
# final_values(f, n; T=Float64) = zeros(T, n)

composite_recorder(args...) = n->merge(map(Base.Fix2(initialize_parts, n), args)...)
initialize_parts(x::Symbol, n) = initialize_parts(Val(x), n)
initialize_parts(::Val{:initial_position}, n) = (;initial_position=InitialRecorder(position; n))
initialize_parts(::Val{:initial_gradient}, n) = (;initial_gradient=InitialRecorder(gradient; n))
initialize_parts(::Val{:initial_lpdf}, n) = (;initial_lpdf=InitialRecorder(lpdf; n))
initialize_parts(::Val{:positions}, n) = (;positions=ContinuousRecorder(position; n))
initialize_parts(::Val{:gradients}, n) = (;gradients=ContinuousRecorder(gradient; n))
initialize_parts(::Val{:lpdfs}, n) = (;lpdfs=ContinuousRecorder(lpdf; n))
initialize_parts(::Val{:position_jumps}, n) = merge(
    initialize_parts(:initial_position, n),
    initialize_parts(:positions, n),
)
initialize_parts(::Val{:gradient_jumps}, n) = merge(
    initialize_parts(:initial_gradient, n),
    initialize_parts(:gradients, n),
)
initialize_parts(::Val{:lpdf_jumps}, n) = merge(
    initialize_parts(:initial_lpdf, n),
    initialize_parts(:lpdfs, n),
)
initialize_parts(::Val{:everything}, n) = merge(
    initialize_parts(:position_jumps, n),
    initialize_parts(:gradient_jumps, n),
    initialize_parts(:lpdf_jumps, n),
)