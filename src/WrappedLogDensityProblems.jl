abstract type WrappedLogDensityProblem{P} end
LogDensityProblems.capabilities(::Type{<:WrappedLogDensityProblem{P}}) where {P} = LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::WrappedLogDensityProblem) = LogDensityProblems.dimension(parent(p))
LogDensityProblems.logdensity(p::WrappedLogDensityProblem, x) = LogDensityProblems.logdensity(parent(p), x)
LogDensityProblems.logdensity_and_gradient(p::WrappedLogDensityProblem, x) = LogDensityProblems.logdensity_and_gradient(parent(p), x)
Base.show(io::IO, p::WrappedLogDensityProblem) = print(io, typeof(p).name.wrapper, "(", parent(p), ")") 


struct NamedPosterior{P} <: WrappedLogDensityProblem{P}
    posterior::P
    name::String
end
Base.parent(p::NamedPosterior) = p.posterior
Base.show(io::IO, p::NamedPosterior) = print(io, p.name) 

struct CountingPosterior{P} <: WrappedLogDensityProblem{P}
    posterior::P
    count::Ref{Int64}
    CountingPosterior(p) = new{typeof(p)}(p, Ref(0))
end
Base.parent(p::CountingPosterior) = p.posterior
LogDensityProblems.logdensity_and_gradient(p::CountingPosterior, x) = begin
    p.count[] += 1
    LogDensityProblems.logdensity_and_gradient(parent(p), x)
end

struct RecordingPosterior2{P,T,R,G} <: WrappedLogDensityProblem{P}
    posterior::P
    halo_position::ElasticMatrix{T,Vector{T}}
    halo_gradient::ElasticMatrix{T,Vector{T}}
    posterior_position::ElasticMatrix{T,Vector{T}}
    posterior_gradient::ElasticMatrix{T,Vector{T}}
    recorder::R
    rng::G
end
RecordingPosterior2(p; rng, recorder=log(1e-2)) = begin
    n = LogDensityProblems.dimension(p) 
    RecordingPosterior2(
        p, 
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0), 
        recorder,
        rng
    )
end
Base.parent(p::RecordingPosterior2) = p.posterior
record!(p::RecordingPosterior2, z; is_initial, dH) = begin 
    if !is_initial && dH > log(1e-2)
        append!(p.halo_position, z.Q.q)
        append!(p.halo_gradient, z.Q.∇ℓq)
    end

end
function DynamicHMC.leaf(trajectory::DynamicHMC.TrajectoryNUTS{DynamicHMC.Hamiltonian{K,P}}, z, is_initial) where {K, P<:RecordingPosterior2}
    (;H, π₀, min_Δ, turn_statistic_configuration) = trajectory
    p = H.ℓ
    Δ = is_initial ? zero(π₀) : DynamicHMC.logdensity(H, z) - π₀
    record!(p, z; is_initial, dH=Δ)
    isdiv = Δ < min_Δ
    v = DynamicHMC.leaf_acceptance_statistic(Δ, is_initial)
    if isdiv
        nothing, v
    else
        τ = DynamicHMC.leaf_turn_statistic(turn_statistic_configuration, H, z)
        (z, Δ, τ), v
    end
end
mutable struct LimitedRecorder2
    target::Int64
    thin::Int64
    outer_count::Int64
    inner_count::Int64
    triggered::Bool
    written::Bool
end
LimitedRecorder2(target, thin) = LimitedRecorder2(target, thin, 1, 0, false, false)
LimitedRecordingPosterior3{P,T,G} = RecordingPosterior2{P,T,LimitedRecorder2,G}
record!(p::LimitedRecordingPosterior3, z; is_initial, dH) = begin 
    r = p.recorder::LimitedRecorder2
    if !r.triggered
        if !is_initial && dH > log(1e-2)
            r.written = true
            if size(p.halo_position, 2) < r.outer_count
                append!(p.halo_position, z.Q.q)
                append!(p.halo_gradient, z.Q.∇ℓq)
            else
                p.halo_position[:, r.outer_count] .= z.Q.q
                p.halo_gradient[:, r.outer_count] .= z.Q.∇ℓq
            end 
            r.triggered = rand(p.rng) <= 1/(r.thin-r.inner_count)
        end
    end
    r.inner_count += 1
    if r.inner_count == r.thin
        r.written && (r.outer_count = 1 + (r.outer_count % r.target))
        r.inner_count = 0
        r.triggered = false
        r.written = false
    end

end
reset!(x::ElasticArray) = resize!(x, Base.front(size(x))..., 0)
reset!(p::RecordingPosterior2) = begin 
    map(reset!, (p.halo_position, p.halo_gradient, p.posterior_position, p.posterior_gradient))
    reset!(p.recorder)
end
reset!(r::LimitedRecorder2) = begin 
    r.outer_count = 1
    r.inner_count = 0
    r.triggered = false
    r.written = false
end
