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

"""
    count_and_time(f, problem) -> (; elapsed, n_evaluations, result)

Wrap `problem` in a fresh `CountingPosterior`, time the call to `f(wrapped)`,
and return the elapsed wall-clock seconds, the total number of
`logdensity_and_gradient` evaluations, and the closure's return value.

Designed for `do`-block syntax so a sampler call site reads as:

```julia
(; elapsed, n_evaluations, result) = count_and_time(problem) do cp
    sampler_call(rng, cp; ...)
end
```
"""
function count_and_time(f, problem)
    cp = CountingPosterior(problem)
    t0 = time()
    result = f(cp)
    (elapsed = time() - t0, n_evaluations = cp.count[], result)
end

"""
    RecordingPosterior2(posterior; rng, recorder=nothing)

The log density the sampler actually runs against during warm-up: `posterior`,
plus the four matrices adaptation reads from.

`posterior_position` / `posterior_gradient` collect the accepted draws.
`halo_position` / `halo_gradient` collect the **halo** — one intermediate state
per NUTS trajectory, sampled from the exact marginal proposal probabilities over
that trajectory's leaves (`sample_leaf`, called by `finalize_leaf_recording!`),
which is what both the linear-transformation selection and
`find_reparametrization!` are fitted on. A `LimitedRecorder2` in `recorder`
caps the halo at `recording_target` columns by overwriting in a ring.

`reset!` empties all four and is what a restarting warm-up window does.
"""
struct RecordingPosterior2{P,T,L,R,G} <: WrappedLogDensityProblem{P}
    posterior::P
    halo_position::ElasticMatrix{T,Vector{T}}
    halo_gradient::ElasticMatrix{T,Vector{T}}
    posterior_position::ElasticMatrix{T,Vector{T}}
    posterior_gradient::ElasticMatrix{T,Vector{T}}
    leaves::L
    recorder::R
    rng::G
end
RecordingPosterior2(p; rng, recorder=nothing) = begin
    n = LogDensityProblems.dimension(p) 
    RecordingPosterior2(
        p, 
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        ElasticMatrix{Float64,Vector{Float64}}(undef, n, 0),
        NUTSLeaves(n),
        recorder,
        rng
    )
end
Base.parent(p::RecordingPosterior2) = p.posterior
function DynamicHMC.leaf(trajectory::DynamicHMC.TrajectoryNUTS{DynamicHMC.Hamiltonian{K,P}}, z, is_initial) where {K, P<:RecordingPosterior2}
    (;H, π₀, min_Δ, turn_statistic_configuration) = trajectory
    p = H.ℓ
    Δ = is_initial ? zero(π₀) : DynamicHMC.logdensity(H, z) - π₀
    record_leaf!(p.leaves, z, Δ)
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
    # The remaining fields retain the serialized layout used by checkpoints
    # written before acceptance-weighted recording. Only `outer_count` remains
    # active: it is the next ring-buffer destination.
    thin::Int64
    outer_count::Int64
    inner_count::Int64
    triggered::Bool
    written::Bool
end
LimitedRecorder2(target) = target > 0 ? LimitedRecorder2(target, 0, 1, 0, false, false) :
    throw(ArgumentError("recording_target must be positive, got $target"))
LimitedRecordingPosterior3{P,T,L,G} = RecordingPosterior2{P,T,L,LimitedRecorder2,G}

function _store_leaf!(p::RecordingPosterior2, leaf_index, destination)
    if size(p.halo_position, 2) < destination
        append!(p.halo_position, @view p.leaves.position[:, leaf_index])
        append!(p.halo_gradient, @view p.leaves.gradient[:, leaf_index])
    else
        p.halo_position[:, destination] .= @view p.leaves.position[:, leaf_index]
        p.halo_gradient[:, destination] .= @view p.leaves.gradient[:, leaf_index]
    end
end

function record_weighted_leaf!(p::RecordingPosterior2)
    leaf_index = sample_leaf(p.rng, p.leaves)
    destination = size(p.halo_position, 2) + 1
    _store_leaf!(p, leaf_index, destination)
    p
end

function record_weighted_leaf!(p::LimitedRecordingPosterior3)
    recorder = p.recorder
    leaf_index = sample_leaf(p.rng, p.leaves)
    _store_leaf!(p, leaf_index, recorder.outer_count)
    recorder.outer_count = 1 + (recorder.outer_count % recorder.target)
    p
end

function finalize_leaf_recording!(p::RecordingPosterior2, depth)
    finalize_leaf_weights!(p.leaves, depth)
    record_weighted_leaf!(p)
end

reset!(x::ElasticArray) = resize!(x, Base.front(size(x))..., 0)
reset!(p::RecordingPosterior2) = begin 
    map(reset!, (p.halo_position, p.halo_gradient, p.posterior_position, p.posterior_gradient))
    reset!(p.leaves)
    reset!(p.recorder)
end
reset!(::Nothing) = nothing
reset!(r::LimitedRecorder2) = begin 
    r.outer_count = 1
    r.inner_count = 0
    r.triggered = false
    r.written = false
end
