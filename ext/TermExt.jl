module TermExt
import Term, WarmupHMC

ProgressBar{I} = WarmupHMC.Progress{Term.Progress.ProgressBar, I}
ProgressJob{I} = WarmupHMC.Progress{Term.Progress.ProgressJob, I}

import Term.Progress: AbstractColumn, DescriptionColumn, CompletedColumn, SeparatorColumn, ProgressColumn
struct StringColumn <: AbstractColumn
    parent::Term.Progress.ProgressJob
    msg::Ref{String}
    StringColumn(job::Term.Progress.ProgressJob, msg::AbstractString="             ") = new(job, Ref(msg))
end 
Base.getproperty(x::StringColumn, k::Symbol) = if hasfield(typeof(x), k)
    getfield(x, k)
else
    @assert k == :measure
    Term.Progress.Measure(x.msg[])
end
Term.Progress.update!(col::StringColumn, color::String) = Term.Segment(col.msg[], color).text

function renderloop(pbar, lock)
    while pbar.running
        Base.lock(lock) do 
            Term.Progress.render(pbar)
        end
        sleep(pbar.Δt)
    end
end
owner(bar::ProgressBar) = bar
owner(job::ProgressJob) = job.info.owner
root(bar::ProgressBar) = bar
root(job::ProgressJob) = root(owner(job))
level(bar::ProgressBar) = 0
level(job::ProgressJob) = job.info.level
priority(job::ProgressJob) = job.info.priority
pos(bar::ProgressBar) = tuple()
pos(job::ProgressJob) = job.info.pos
owns(x::WarmupHMC.Progress) = x.info.owns
labels(x::WarmupHMC.Progress) = x.info.labels
max_level(x::WarmupHMC.Progress) = get(root(x).info, :max_level, +Inf) 
max_rows(x::WarmupHMC.Progress) = get(root(x).info, :max_rows, 32)
id(x::ProgressJob) = parent(x).id
propagates(x::WarmupHMC.Progress) = get(x.info, :propagates, false)
# isactive(x::WarmupHMC.Progress) = x.info.isactive[]
# queue(x::ProgressBar) = x.info.queue

# addchild!(x::Term.Progress.ProgressBar; owner=nothing, lock=ReentrantLock(), kwargs...) = addchild!(
#     owner, WarmupHMC.Progress(x, (;lock, next_id=Ref(1), priority=[], pos=[], kwargs...))
# )

# addchild!(x::Term.Progress.ProgressJob; owner=nothing, lock=ReentrantLock(), kwargs...) = addchild!(
#     owner, WarmupHMC.Progress(x, (;owner, owns=Vector(), labels=Dict(), lock, isactive=Ref(true), kwargs...))
# )
# addchild!(::Nothing, x::WarmupHMC.Progress) = x
# # find_last_active(owner::WarmupHMC.Progress) = if !isactive(owner)
# #     nothing
# # else
# #     for x in Iterators.reverse(owns(owner))
# #         rv = find_last_active(x)
# #         isnothing(rv) || return rv
# #     end
# #     return owner
# # end
# addchild!(owner::ProgressBar, x::ProgressJob) = lock(owner) do 
#     pbar = parent(owner)
#     job = parent(x)
#     push!(pbar.jobs, job)
# end
# addchild!(owner::ProgressJob, x::ProgressJob) = lock(owner) do 
#     push!(owns(owner), x)
#     x
# end
always_true(x) = true
childfilter(owner::WarmupHMC.Progress) = get(root(owner).info, :filter, always_true)
acceptschild(owner::WarmupHMC.Progress, x::WarmupHMC.Progress) = level(owner) < max_level(owner) && length(parent(root(owner)).jobs) < max_rows(owner) && childfilter(owner)(x)
Base.lock(f, x::WarmupHMC.Progress) = lock(f, x.info.lock)
insertsorted!(v, x; by=identity) = insert!(v, searchsortedfirst(v, x; by), x)

WarmupHMC.Progress(::Type{Term.Progress.ProgressBar}; kwargs...) = WarmupHMC.Progress(Val(Term.Progress.ProgressBar), kwargs)
WarmupHMC.initialize_progress!(p::WarmupHMC.Progress{Val{Term.Progress.ProgressBar}}; kwargs...) = WarmupHMC.initialize_progress!(
    Term.Progress.ProgressBar;
    p.info..., kwargs...
)
WarmupHMC.initialize_progress!(p::WarmupHMC.Progress{Val{Term.Progress.ProgressBar}}, N; kwargs...) = WarmupHMC.initialize_progress!(
    Term.Progress.ProgressBar, N;
    p.info..., kwargs...
)
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}, N; description="Running...", kwargs...) = WarmupHMC.initialize_progress!(
    WarmupHMC.initialize_progress!(Term.Progress.ProgressBar; kwargs...); 
    N, description, propagates=true
)
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}; width=120, kwargs...) = begin 
    bar = Term.Progress.ProgressBar(;width, columns=[DescriptionColumn, CompletedColumn, SeparatorColumn, ProgressColumn])
    Term.Progress.start!(bar)
    lock = ReentrantLock()
    thread = Threads.@spawn renderloop(bar, lock)
    WarmupHMC.Progress(bar, (;
        thread, 
        lock, 
        current_id=Ref(0), 
        owns=[], 
        priority=[], 
        pos=[], 
        kwargs...
    ))
end
WarmupHMC.initialize_progress!(owner::WarmupHMC.Progress, N; kwargs...) = WarmupHMC.initialize_progress!(owner; N, kwargs...)
WarmupHMC.initialize_progress!(owner::WarmupHMC.Progress; N=nothing, description, key=nothing, value="", transient=false, kwargs...) = begin
    bar = root(owner)
    pbar = parent(bar)
    jid = lock(bar) do 
        bar.info.current_id[] += 1
        bar.info.current_id[]
    end
    pjob = Term.ProgressJob(jid, N, description, pbar.columns, pbar.width, pbar.columns_kwargs, transient)
    # Initialize columns
    if isnothing(N)
        pjob.columns = [DescriptionColumn(pjob), StringColumn(pjob, value)]
    else
        pjob.columns = [DescriptionColumn(pjob), CompletedColumn(pjob), SeparatorColumn(pjob), ProgressColumn(pjob), StringColumn(pjob, "")]
    end
    # Apply color/style
    Term.Progress.render(pjob, pbar)
    job = lock(owner) do
        owner.info.current_id[] += 1
        job = WarmupHMC.Progress(pjob, (;
            owner, 
            lock=ReentrantLock(), 
            current_id=Ref(0), 
            owns=[], 
            labels=Dict(),
            level=level(owner)+1,
            priority=(level(owner)+1, owner.info.current_id[]),
            pos=(pos(owner)..., owner.info.current_id[]), 
            running=Ref(true),
            kwargs...
        )) 
        push!(owns(owner), job)
        job 
    end
    lock(bar) do 
        insertsorted!(bar.info.priority, job; by=priority)
        insertsorted!(bar.info.pos, job; by=pos)
        recomputejobs!(bar)
    end
    job
end
persists(job::ProgressJob) = job.info.running[] || !parent(job).transient
recomputejobs!(bar::ProgressBar) = lock(bar) do 
    pbar = parent(bar)
    filter!(persists, bar.info.pos)
    filter!(persists, bar.info.priority)
    append!(
        empty!(pbar.jobs), 
        parent.(
            getindex.(
                Ref(bar.info.pos),
                sort!(getindex.(
                    Ref(Dict(zip(id.(bar.info.pos), eachindex(bar.info.pos)))), 
                    id.(bar.info.priority[1:min(length(bar.info.priority), max_rows(bar))])
                ))
            )
        )
    )
end

WarmupHMC.update_progress!(job::Term.Progress.ProgressJob, i::Integer) = if isnothing(job.N)
    WarmupHMC.update_progress!(job, WarmupHMC.short_string(i))
else
    job.i = i
end
WarmupHMC.update_progress!(::Term.Progress.ProgressJob, ::Nothing) = nothing
WarmupHMC.update_progress!(job::Term.Progress.ProgressJob, value) = job.columns[end].msg[] = WarmupHMC.short_string(value)
WarmupHMC.update_progress!(job::ProgressJob, i=parent(job).i+1; kwargs...) = begin 
    WarmupHMC.update_progress!(parent(job), i)
    for (key, value) in pairs(kwargs)
        sjob = get!(labels(job), key) do
            skey = rpad(string(key), maximum(length ∘ string, keys(kwargs)))
            WarmupHMC.initialize_progress!(job; key, description="$skey:")
        end
        WarmupHMC.update_progress!(sjob, value)
    end
    WarmupHMC.update_progress!(owner(job), nothing)
end
WarmupHMC.update_progress!(bar::ProgressBar, ::Nothing) = yield()
inner_finalize_progress!(job::ProgressJob; transient=false) = if job.info.running[] || parent(job).transient == transient
    transient = parent(job).transient |= transient 
    job.info.running[] = false
    for sjob in owns(job)
        inner_finalize_progress!(sjob; transient)
    end
end
WarmupHMC.finalize_progress!(job::ProgressJob) = lock(root(job)) do
    inner_finalize_progress!(job)
    recomputejobs!(root(job))
    propagates(job) && WarmupHMC.finalize_progress!(owner(job))
end
WarmupHMC.finalize_progress!(bar::ProgressBar) = lock(bar) do
    pbar = parent(bar) 
    Term.Progress.render(pbar)
    Term.Progress.stop!(pbar)
    yield()
end

WarmupHMC.pathfinder_callback(job::ProgressJob) = (state, args...) -> (WarmupHMC.update_progress!(job, state.iter); false)

end