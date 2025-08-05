# import WarmupHMC: with_progress, ensurevector, initialize_mcmc, DynamicHMC, MatrixFactorization, MatrixInverse, SuccessiveReflections, RecordingPosterior2, LimitedRecorder2, update_loss!
# import Serialization: serialize, deserialize
using Random, DataFrames


myprogress(x; kwargs...) = x
myprogress(x::Symbol; parent) = x == :__progress__ ? parent : x
myprogress(x::Expr; parent=:(default_progress()), description=nothing) = if x.head == :for
    h, b = x.args
    @assert Meta.isexpr(h, :(=))
    lhs, rhs = h.args
    @assert Meta.isexpr(b, :block)
    p = gensym("progress")
    it = gensym("it")
    description = something(description, "for $h")
    h = Expr(:(=),lhs, it)
    b = Expr(:block, myprogress.(b.args; parent=p)..., :(update_progress!($p)))
    x = Expr(:for, h, b)
    quote 
        $it = $(rhs)
        with_progress($parent, length($it); description=$description, transient=true) do $p
            @sync $x
        end
    end
elseif x.head == :while
    h, b = x.args
    @assert Meta.isexpr(h, :call) && h.args[1] == :<
    _, lhs, rhs = h.args
    @assert Meta.isexpr(b, :block)
    p = gensym("progress")
    it = gensym("it")
    description = something(description, "while $h")
    b = Expr(:block, myprogress.(b.args; parent=p)..., :(update_progress!($p, $lhs)))
    x = Expr(:while, h, b)
    quote 
        $it = $(rhs)
        with_progress($parent, $rhs; description=$description, transient=true) do $p
            @sync $x
        end
    end
elseif x.head == :do
    h, b = x.args
    @assert Meta.isexpr(h, :call)
    g, args... = h.args
    @assert Meta.isexpr(b, :(->))
    args = Meta.isexpr(get(args, 1, nothing), :parameters) ? (args[1], b, args[2:end]...) :  (b, args...)
    myprogress(Expr(:call, g, args...); parent, description)
elseif x.head == :call
    f, args... = x.args
    if f in (:map, :pmap)
        g, args... = args
        p = gensym("progress")
        it = gensym("it")
        tmp = gensym("tmp")
        g = quote 
            $tmp = $(myprogress(g; parent=p))(args...)
            update_progress!($p)
            $tmp
        end
        f == :pmap && (g = :(Threads.@spawn $g))
        g = :((args...)->$g)
        x = Expr(:call, :map, g, it, args[2:end]...)
        description = something(description, "$f($(args[1]), ...) do ...")
        return quote 
            $it = $(args[1])
            with_progress($parent, length($it); description=$description, transient=true) do $p
                map(fetch, $x)
            end
            
        end
    else
        Expr(x.head, myprogress.(x.args; parent)...)
    end
elseif x.head == :macrocall && x.args[1] == Symbol("@myprogress")
    myprogress(x.args[4]; parent, description=x.args[3])
elseif x.head == :macrocall && x.args[1] == Symbol("@noprogress")
    x = x.args[3]
    Expr(x.head, myprogress.(x.args; parent)...)
else
    Expr(x.head, myprogress.(x.args; parent)...)
end
macro myprogress(x)
    esc(myprogress(x))
end
macro myprogress(parent, x)
    esc(myprogress(x; parent))
end
macro myprogress(parent, description, x)
    esc(myprogress(x; parent, description))
end


full_initialize_mcmc(lpdf, init; rng, progress, kwargs...) = begin 
    (;position, squared_scale) = initialize_mcmc(lpdf, init; rng, progress, kwargs...)
    position_and_gradient = DynamicHMC.evaluate_ℓ(lpdf, position; strict=true)
    energy = scale_to_energy(factorize(squared_scale).L)
    stepsize = DynamicHMC.find_initial_stepsize(
        DynamicHMC.InitialStepsizeSearch(), 
        DynamicHMC.local_log_acceptance_ratio(
            DynamicHMC.Hamiltonian(energy, lpdf),
            DynamicHMC.PhasePoint(position_and_gradient, DynamicHMC.rand_p(rng, energy))
        )
    )
    return (;position_and_gradient, energy, stepsize)
end

updatestate(affected, x::Expr) = begin 
    affected = ensure_nt(affected)
    @assert x.head == :(=)
    lhs, rhs = x.args
    @assert isa(lhs, Symbol)
    qlhs = QuoteNode(lhs)
    postfix = "$(lhs) and $(affected)."
    quote 
        if hasfield(typeof(state), $qlhs)
            @info "Restoring " * $postfix
        else
            @info "Updating " * $postfix
            state = merge(state, (;$lhs=$rhs), $affected)
            save_state(state)
        end
        (;$lhs) = state
        $affected = state
    end
end
macro updatestate(x)
    esc(updatestate(:((;)), x))
end
macro updatestate(affected, x)
    esc(updatestate(affected, x))
end

ensure_nt(x::Symbol) = :(;$x)
ensure_nt(x::Expr) = begin
    @assert x.head == :tuple 
    @assert Meta.isexpr(x.args[1], :parameters)
    x
end

updateloop(lhs, loop) = begin 
    lhs = ensure_nt(lhs)
    @assert Meta.isexpr(loop, :for)
    head, body = loop.args
    @assert Meta.isexpr(head, :(=))
    it = head.args[1]
    @assert isa(it, Symbol)
    qit = QuoteNode(it)
    slhs = string(lhs)
    Expr(
        :for, head, 
        Expr(
            :block, 
            quote
                if hasproperty(state, $qit)
                    if state.$it >= $it
                        continue
                    else
                        # @info "Restoring " * $slhs * " at " * string($it)
                        $lhs = state
                    end
                end
            end,
            body.args...,
            quote 
                state = merge(state, $lhs, (;$it))
                save_state(state)
            end
        )
    )
end

macro updateloop(lhs, loop)
    esc(updateloop(lhs, loop))
end

donothing(args...; kwargs...) = nothing
initialize_state(::Nothing, ::Nothing) = (;)
# initialize_state(::Nothing, state_path::AbstractString) = initialize_state(isfile(state_path) ? deserialize(state_path) : (;), state_path)
initialize_state(state::NamedTuple, ::Nothing) = state
initialize_state(state::NamedTuple, state_path::AbstractString) = merge(state, (;state_path))
function save_state end
# save_state(state::NamedTuple) = if hasproperty(state, :state_path)
#     @info "Writing to $(state.state_path)!" 
#     serialize(state.state_path, state)
# end
drop_fields(state::NamedTuple, args...) = (;[
    key=>value for (key, value) in pairs(state) if key ∉ args
]...)
getproperties(x, key) = getproperty(x, key)
getproperties(x, key::Int) = getfield(x, key)
getproperties(x, key, args...) = getproperties(getproperties(x, key), args...)
# import WarmupHMC: WrappedLogDensityProblem, ElasticMatrix, update_loss!

struct NUTSPosterior2{P,T} <: WrappedLogDensityProblem{P}
    posterior::P
    position::ElasticMatrix{T,Vector{T}}
    gradient::ElasticMatrix{T,Vector{T}}
    dH::Vector{T}
    idxs::Vector{Int}
end
Base.parent(lpdf::NUTSPosterior2) = lpdf.posterior
NUTSPosterior2(lpdf) = begin
    n = LogDensityProblems.dimension(lpdf)
    NUTSPosterior2(lpdf, ElasticMatrix(zeros((n, 0))), ElasticMatrix(zeros((n, 0))), zeros(0), fill(0, 2))
end
reset!(x::Vector{<:Number}) = empty!(x)
reset!(x::Vector{<:AbstractArray}) = map(reset!, x)
reset!(x::NamedTuple) = map(reset!, x)
reset!(lpdf::NUTSPosterior2) = begin
    map(reset!, (lpdf.position, lpdf.gradient, lpdf.dH))
    lpdf.idxs .= 0
end
DynamicHMC.move(trajectory::DynamicHMC.TrajectoryNUTS{DynamicHMC.Hamiltonian{K,P}}, z, fwd) where {K, P<:NUTSPosterior2} = begin 
    (; H, ϵ) = trajectory
    z = DynamicHMC.leapfrog(H, z, fwd ? ϵ : -ϵ)
    lpdf = H.ℓ
    dH = DynamicHMC.logdensity(H, z) - trajectory.π₀
    f! = fwd ? append! : prepend!
    f!(lpdf.position, z.Q.q)
    f!(lpdf.gradient, z.Q.∇ℓq)
    f!(lpdf.dH, dH)
    lpdf.idxs[fwd ? 2 : 1] += 1
    z
end
abs_idx(lpdf::NUTSPosterior2, x::AbstractVector) = begin 
    idx = findfirst(==(x), eachcol(lpdf.position))
    isnothing(idx) && return 0
    return idx > lpdf.idxs[1] ? idx - lpdf.idxs[1] : lpdf.idxs[1] - idx + 1
end

struct CallbackPosterior{P,F} <: WrappedLogDensityProblem{P}
    func::F
    posterior::P
end
Base.parent(p::CallbackPosterior) = p.posterior
LogDensityProblems.logdensity(p::CallbackPosterior, x) = p.func(
    LogDensityProblems.logdensity,
    x,
    LogDensityProblems.logdensity(parent(p), x)
)
LogDensityProblems.logdensity_and_gradient(p::CallbackPosterior, x) = p.func(
    LogDensityProblems.logdensity_and_gradient,
    x,
    LogDensityProblems.logdensity_and_gradient(parent(p), x)
)
struct OneOfN{T<:Union{Tuple,NamedTuple}}
    parent::T
    active::Int
end
Base.parent(x::OneOfN) = x.parent[x.active]
Base.getindex(x::OneOfN) = parent(x)
scale_to_energy(scale) = DynamicHMC.GaussianKineticEnergy(MatrixFactorization(scale, scale'), MatrixInverse(scale'))
energy_to_scale(energy::DynamicHMC.GaussianKineticEnergy) = energy.M⁻¹.m1
scale_then_reflect(n::Int) = MatrixFactorization(SuccessiveReflections(n), Diagonal(ones(n)))
update_loss!(energy::DynamicHMC.GaussianKineticEnergy, args...; kwargs...) = update_loss!(energy_to_scale(energy), args...; kwargs...)

@views resumable_parallel_warmup_mcmc(
    rngs, lpdf;
    n_outer=10,
    state=nothing, state_path=nothing, 
    stepsize_adaptation_limit=100, 
    target_acceptance_rate=.8, 
    base_recording_target=1_000,
    max_recording_target=100_000,
    init=missing, 
    progress=nothing, 
    callback=donothing,
    kwargs...
) = with_progress(progress, n_outer; description="Parallel sampling... ($(Threads.nthreads()) threads, $n_outer outer iterations)") do progress
    state = initialize_state(state, state_path)
    dimension = LogDensityProblems.dimension(lpdf)
    n_chains = length(rngs)
    start_time = time_ns()
    df = DataFrame(;
        outer_i=fill(0, 0), 
        chain_idx=fill(0, 0), 
        stepsize=zeros(0), 
        acceptance_rate=zeros(0), 
        steps=fill(0, 0), 
        divergent=fill(false, 0),
        turned=fill(false, 0), 
        valid=fill(false, 0), 
        n_records=fill(0, 0),
        finish_time=zeros(0),
        arrival_time=zeros(0),
        max_depth=zeros(0),
        jump=zeros(0),
        log_density=zeros(0)
    )
    plock = ReentrantLock()

    # state = drop_fields(state, :init)
    @updatestate (;rngs, df) pathfinder_init = @myprogress progress "Pathfinder initialization" pmap(1:n_chains) do chain_idx
        rng = rngs[chain_idx]
        clpdf = CallbackPosterior(lpdf) do f, x, g
            finish_time = time_ns() - start_time
            f == LogDensityProblems.logdensity_and_gradient && lock(plock) do 
                row = (;
                    outer_i=0, 
                    chain_idx, 
                    stepsize=NaN, 
                    acceptance_rate=NaN, 
                    steps=1, 
                    divergent=false,
                    turned=false, 
                    valid=false, 
                    n_records=0,
                    finish_time,
                    arrival_time=time_ns() - start_time,
                    max_depth=0,
                    jump=0,
                    log_density=g[1]
                )
                push!(df, row)
            end
            g
        end
        full_initialize_mcmc(clpdf, init; rng, progress=__progress__, kwargs...)
    end
    n_clusters = n_chains+1
    chainwise_state = [merge(pathfinder_init[i], (;cluster_idx=1+i)) for i in 1:n_chains]
    global_state = (;
        # energy=scale_to_energy(scale_then_reflect(dimension)),
        stepsize_regression=ConjugateLinearRegression(2+n_chains),
        grecorder=(;
            halo_position=ElasticMatrix(zeros((dimension, 0))),
            halo_gradient=ElasticMatrix(zeros((dimension, 0))),
            halo_idx=fill(0, 0),
            position=ElasticMatrix(zeros((dimension, 0))),
            position_idx=fill(0, 0),
        ),
        ess=zeros(dimension),
        rhat=zeros(dimension)
    )
    global_state.stepsize_regression.ab .= [3, 1, 0]

    @updateloop (;rngs, df, chainwise_state, global_state) for outer_i in 1:n_outer
        inner_start_time = time_ns()
        (;stepsize_regression, grecorder, ess, rhat) = global_state
        n_rows = size(df, 1)
        n_draws = zeros(n_chains)
        n_transitions = zeros(n_chains)
        n_evals = zeros(n_chains)
        n_clusters = maximum(x->x.cluster_idx, chainwise_state)
        cluster_n_rows = zeros(n_clusters)
        nominal_recording_target = base_recording_target * 2^(outer_i-1)
        recording_target = min(max_recording_target, nominal_recording_target)
        recording_thin = ceil(Int, nominal_recording_target / recording_target)
        n_recorded = 0
        update_progress!(progress, outer_i-1;
            n_clusters,
            rhat,
            ess=sort(ess),
            n_draws=Speeds(n_draws, inner_start_time),
            n_transitions=Speeds(n_transitions, inner_start_time),
            n_evals=Speeds(n_evals, inner_start_time),
        )
        reset!(grecorder)
        
        with_progress(progress, recording_target; description="Recording target", transient=true) do recorded_progress
        @myprogress recorded_progress "Outer iteration $outer_i ($n_chains chains):" pmap(1:n_chains) do chain_idx
            rng = rngs[chain_idx]
            chain_state = chainwise_state[chain_idx]
            (;position_and_gradient, stepsize, cluster_idx, energy) = chain_state
            stepsize_regression_x = zeros(n_clusters+1)
            stepsize_regression_x[cluster_idx] = 1
            recording_lpdf = NUTSPosterior2(lpdf)
            hamiltonian = DynamicHMC.Hamiltonian(energy, recording_lpdf)
            stats = nothing
            max_depth = 0
            pre_n_rows = 0
            
            @noprogress while n_recorded < recording_target
                lock(plock) do
                    pre_n_rows = cluster_n_rows[cluster_idx] 
                    if cluster_n_rows[cluster_idx] <= stepsize_adaptation_limit
                        stepsize = if maybeready(stepsize_regression)
                            (;beta) = rand(rng, stepsize_regression; q=.25)
                            alpha, beta = beta[cluster_idx], beta[end]
                            clamp(exp((target_acceptance_rate - alpha) / beta), .5*stepsize, 2*stepsize)
                        else
                            if isnothing(stats)
                                logjitter(stepsize)
                            else
                                logjitter(sqrt(stats.acceptance_rate > target_acceptance_rate ? 2 : .5) * stepsize; f=sqrt(2))
                            end
                        end
                    else
                        prepare!(stepsize_regression)
                        beta = stepsize_regression.location
                        alpha, beta = beta[cluster_idx], beta[end]
                        stepsize = exp((target_acceptance_rate - alpha) / beta)
                    end
                    max_depth = ceil(Int, log2(2 + (recording_target - n_recorded) / (2 * n_chains)))
                end
                max_depth > 0 || break
                (position_and_gradient, stats) = DynamicHMC.sample_tree(
                    rng, 
                    DynamicHMC.NUTS(;max_depth), 
                    hamiltonian, 
                    position_and_gradient, 
                    stepsize
                )
                finish_time = time_ns() - inner_start_time
                chainwise_state[chain_idx] = merge(chainwise_state[chain_idx], (;position_and_gradient, stepsize))
                valid_records = eachindex(recording_lpdf.dH)[recording_lpdf.dH .> -100]
                valid_records = valid_records[1:recording_thin:end]
                n_records = length(valid_records)
                jump = stepsize * abs_idx(recording_lpdf, position_and_gradient.q)
                log_density = position_and_gradient.ℓq
                turned = stats.termination != DynamicHMC.REACHED_MAX_DEPTH
                divergent = DynamicHMC.is_divergent(stats.termination)
                n_evals[chain_idx] += stats.steps
                n_transitions[chain_idx] += 1
                lock(plock) do
                    cluster_n_rows[cluster_idx] += 1
                    append!(grecorder.halo_position, recording_lpdf.position[:, valid_records])
                    append!(grecorder.halo_gradient, recording_lpdf.gradient[:, valid_records])
                    append!(grecorder.halo_idx, fill(chain_idx, n_records))
                    reset!(recording_lpdf)
                    valid = turned && pre_n_rows > stepsize_adaptation_limit
                    if valid
                        append!(grecorder.position, position_and_gradient.q)
                        append!(grecorder.position_idx, chain_idx)
                        n_draws[chain_idx] += 1
                    end
                    if cluster_n_rows[cluster_idx] <= stepsize_adaptation_limit
                        stepsize_regression_x[end] = log(stepsize) 
                        condition!(stepsize_regression, stepsize_regression_x', stats.acceptance_rate)
                    else
                        prepare!(stepsize_regression)
                        beta = stepsize_regression.location
                        alpha, beta = beta[cluster_idx], beta[end]
                        stepsize = exp((target_acceptance_rate - alpha) / beta)
                    end
                    row = (;
                        outer_i, 
                        chain_idx, 
                        stepsize, 
                        stats.acceptance_rate, 
                        stats.steps, 
                        divergent,
                        turned,
                        valid,
                        n_records, 
                        finish_time, 
                        arrival_time=time_ns() - inner_start_time,
                        max_depth, 
                        jump,
                        log_density
                    )
                    push!(df, row)
                    n_recorded = length(grecorder.halo_idx)
                    update_progress!(recorded_progress, n_recorded)
                    update_progress!(progress, outer_i-1;
                        # ess=s(ess, inner_start_time),
                        n_draws=Speeds(n_draws, inner_start_time),
                        n_transitions=Speeds(n_transitions, inner_start_time),
                        n_evals=Speeds(n_evals, inner_start_time),
                    )
                end
                yield()
            end
        end
        end
        (;halo_position, halo_gradient, halo_idx, position, position_idx) = grecorder
        ndf = df[1+n_rows:end, :]
        ld = ndf.log_density
        med_ld = median(ld)
        mad_ld = median(abs.(ld .- med_ld))
        low_ld, high_ld = med_ld .+ (-2,+2) .* mad_ld
        chain_row_idxs = groupedby_idxs(ndf.chain_idx; uy=1:n_chains) 
        cluster_idxs = map(1:n_chains) do chain_idx
            ld = ndf.log_density[chain_row_idxs[chain_idx]]
            ld_regression = ConjugateLinearRegression(eachindex(ld), ld)
            if !maybeready(ld_regression) || (
                .5 * length(ld) < ipred(ld_regression, low_ld) < ipred(ld_regression, med_ld)  
            ) || (
                .5 * length(ld) > ipred(ld_regression, low_ld) > ipred(ld_regression, med_ld)  
            )
                chainwise_state[chain_idx].cluster_idx
            else
                1
            end
        end |> (x->unique_renumber(vcat(1, x))[2:end])
        n_clusters = maximum(cluster_idxs)
        cluster_idxss = groupedby_idxs(cluster_idxs; uy=1:n_clusters)
        chain_halo_idxs = gview(eachindex(halo_idx), groupedby_idxs(halo_idx; uy=1:n_chains))
        energies = map(cluster_idxss) do chain_idxs
            energy = scale_to_energy(scale_then_reflect(dimension))
            idxs = reduce(vcat, chain_halo_idxs[chain_idxs])
            length(idxs) > 10 || return chainwise_state[chain_idxs[1]].energy
            p = halo_position[:, idxs]
            g = halo_gradient[:, idxs]
            update_loss!(energy, copy(p), copy(g); v_f=grad_cov_ev2)
            energy
        end
        chainwise_state = [
            merge(chain_state, (;cluster_idx, energy=energies[cluster_idx])) 
            for (chain_state, cluster_idx, ) in zip(chainwise_state, cluster_idxs)
        ]
        new_stepsize_regression = ConjugateLinearRegression(n_clusters+1)
        pot, prec = stepsize_regression.potential[end], stepsize_regression.precision[end, end]
        new_stepsize_regression.potential[n_clusters+1] = pot / sqrt(prec)
        new_stepsize_regression.precision[n_clusters+1, n_clusters+1] = sqrt(prec)
        new_stepsize_regression.ab[1:2] .= stepsize_regression.ab[1:2]
        stepsize_regression = new_stepsize_regression
        bulk_chain_idxs = cluster_idxss[1]
        draw_idxs = groupedby_idxs(position_idx; uy=1:n_chains)[bulk_chain_idxs]
        sp = sortperm(draw_idxs; by=length, rev=true)
        n_used = argmax(sp) do n_used
            n_used_draws = length(draw_idxs[sp[n_used]])
            (n_used_draws > 10, n_used * n_used_draws)
        end
        n_used_draws = length(draw_idxs[sp[n_used]])
        if n_used_draws >= 10
            draws = zeros((n_used_draws, n_used, dimension))
            for i in 1:n_used
                draws[:, i, :] .= position[:, draw_idxs[sp[i]][end-n_used_draws+1:end]]'
            end
            (;ess, rhat) = MCMCDiagnosticTools.ess_rhat(draws)
        end
        rhat = sort(rhat; rev=true)
        update_progress!(progress, outer_i;
            n_clusters,
            rhat,
            ess=sort(ess),
            n_draws=Speeds(n_draws, inner_start_time),
            n_transitions=Speeds(n_transitions, inner_start_time),
            n_evals=Speeds(n_evals, inner_start_time),
        )
        global_state = (;stepsize_regression, grecorder, ess, rhat)
        callback((;df, grecorder, outer_i))
    end
    callback((;df, global_state.grecorder))
    state
end
unique_renumber(x) = begin 
    ux = unique(x)
    getindex.(Ref(Dict(zip(ux, eachindex(ux)))), x)
end
groupedby_idxs(y; uy=unique(y)) = map(uy) do yi
    filter(eachindex(y)) do i
        y[i] == yi
    end
end
gview(x, idxs) = view.(Ref(x), idxs)
resumable_parallel_warmup_mcmc(callback, rngs, lpdf; kwargs...) = resumable_parallel_warmup_mcmc(rngs, lpdf; callback, kwargs...)


grad_cov_ev2(p, g) = begin
    n = size(g, 1)
    # scales = sqrt.(std.(eachrow(p)) ./ std.(eachrow(g)))
    # sg = scales .* scales' .* cov(g')
    # normalize!(eigen(Symmetric(cov(g')), n:n).vectors[:, 1])
    eigen(Symmetric(cov(p')), 1:1).vectors[:, 1]
end

logjitter(x; f=2.) = x * exp(rand(Uniform(-log(f), +log(f))))

using Distributions


struct ConjugateLinearRegression{T}
    potential::Vector{T}
    precision::Matrix{T}
    ab::Vector{T}
    location::Vector{T}
    L::LowerTriangular{T, Matrix{T}}
end
ConjugateLinearRegression(n; ) = ConjugateLinearRegression(zeros(n), zeros((n,n)), zeros(3), zeros(n), LowerTriangular(zeros((n,n))))
ConjugateLinearRegression(;potential, precision, a, b) = ConjugateLinearRegression(potential, precision, [a, b, 0.], 0*potential, LowerTriangular(0*precision))
ConjugateLinearRegression(x::AbstractVector, y::AbstractVector) = ConjugateLinearRegression(hcat(ones(length(x)), x), y)
ConjugateLinearRegression(X::AbstractMatrix, y::AbstractVector) = condition!(ConjugateLinearRegression(size(X, 2)), X, y)
reset!(p::ConjugateLinearRegression) = begin 
    p.location .= 0
    p.precision .= 0
    p.ab .= 0
    p.L .= 0
    p
end
condition!(p::ConjugateLinearRegression, X, y) = begin
    (n, o) = size(X)
    @assert n == length(y)
    @assert o == length(p.location)
    p.potential .+= X' * y
    p.precision .+= X' * X
    p.ab[1] += n/2
    p.ab[2] += .5 * sum(abs2, y)
    p.ab[3] = 0.
    p
end
prepare!(p::ConjugateLinearRegression) = if p.ab[3] == 0
    parent(p.L) .= cholesky(p.precision).L
    ldiv!(p.location, p.L, p.potential)
    p.ab[3] = p.ab[2] - .5 * sum(abs2, p.location)
    ldiv!(p.L', p.location)
    p
else
    p
end
maybeready(p::ConjugateLinearRegression) = p.ab[3] != 0 || isposdef(p.precision)
Base.rand(rng::AbstractRNG, p::ConjugateLinearRegression, args::Int...; q=0) = begin 
    prepare!(p)
    obs_var_dist = InverseGamma(p.ab[1], p.ab[3])
    obs_var = quantile.(Ref(obs_var_dist), rand(rng, Uniform(q, 1-q), args...))
    # obs_var .= rand(rng, InverseGamma(p.ab[1], p.ab[3]), args...)
    # beta = rand(rng, Uniform(q, 1-q), length(p.potential), args...)
    # beta .= quantile.(Ref(Normal()), beta)
    beta = sqrt.(obs_var') .* quantile.(Ref(Normal()), rand(rng, Uniform(q, 1-q), length(p.potential), args...))
    ldiv!(p.L', beta)
    beta .+= p.location
    (;obs_var, beta)
end 
pred(p::ConjugateLinearRegression, x::Number) = pred(p, [1., x])
pred(p::ConjugateLinearRegression, x::AbstractVector) = begin
    prepare!(p)
    dot(p.location, x)
end
ipred(p::ConjugateLinearRegression, y::Number) = begin 
    prepare!(p)
    alpha, beta = p.location
    (y - alpha) / beta
end

# import WarmupHMC: OnlineStatsBase

running_variance(x::AbstractVector) = begin 
    acc = OnlineStatsBase.Variance()
    [
        (OnlineStatsBase.fit!(acc, xi); var(acc)) for xi in x
    ]
end

running_mean(x::AbstractVector; rate=.01) = begin 
    acc = OnlineStatsBase.Mean(weight=OnlineStatsBase.ExponentialWeight(rate))
    [
        (isfinite(xi) && OnlineStatsBase.fit!(acc, xi); mean(acc)) for xi in x
    ]
end

acceptance_rate_plot(args...; kwargs...) = acceptance_rate_plot!(plot(), args...; kwargs...)
acceptance_rate_plot!(p, stepsize::AbstractVector{<:AbstractVector}, ar::AbstractVector{<:AbstractVector}; kwargs...) = begin 
    for i in eachindex(stepsize)
        acceptance_rate_plot!(p, stepsize[i], ar[i]; color=i, kwargs...)
    end
    p
end
@views acceptance_rate_plot!(p, stepsize, ar; bins=range(0, 1, 11)[2:end-1], kwargs...) = begin 
    n = length(stepsize)
    n_bins = length(bins)
    qs = zeros((n, n_bins))
    ss = sortperm(stepsize; rev=true)
    sar = sortperm(ar)
    weights = 1. .* (stepsize[sar] .== stepsize[ss[1]])
    cweights = zeros(n)
    for i in 2:n
        weights .= exp(-16*abs(stepsize[ss[i]] - stepsize[ss[i-1]])) .* weights .+ (stepsize[ss[i]] .== stepsize[sar])
        cumsum!(cweights, weights)
        cweights ./= cweights[end] 
        for j in eachindex(bins)
            qs[i, j] = ar[sar[searchsortedfirst(cweights, bins[j])]]
        end
    end
    for i in 1:n_bins÷2
        plot!(p, stepsize[ss], qs[:, i], fillrange=qs[:, end-i+1]; alpha=0, fillalpha=2/n_bins, kwargs...)
    end 
    plot!(p, stepsize[ss], qs; kwargs...)
end
@views parallel_ess_rhat(position, idx) = begin
    dimension = size(position, 1) 
    cidxs = groupedby_idxs(idx)
    n_chains = length(cidxs)
    sp = sortperm(cidxs; by=length, rev=true)
    # display(sp')
    map(1:n_chains) do n_used
        n_draws = length(cidxs[sp[n_used]])
        draws = zeros((n_draws, n_used, dimension))
        for i in 1:n_used
            draws[:, i, :] .= position[:, cidxs[sp[i]][end-n_draws+1:end]]'
        end
        MCMCDiagnosticTools.ess_rhat(draws)
    end
end