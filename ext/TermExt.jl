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

function renderloop(pbar)
    while pbar.running
        pbar.paused || Term.Progress.render(pbar)
        sleep(pbar.Δt)
    end
end
owner(bar::ProgressBar) = bar
owner(job::ProgressJob) = job.info.owner
root(bar::ProgressBar) = bar
root(job::ProgressJob) = root(owner(job))
level(bar::ProgressBar) = 0
level(job::ProgressJob) = 1+level(owner(job))
owns(x::WarmupHMC.Progress) = x.info.owns
labels(x::WarmupHMC.Progress) = x.info.labels
max_level(x::WarmupHMC.Progress) = get(root(x).info, :max_level, +Inf) 
propagates(x::WarmupHMC.Progress) = get(x.info, :propagates, false)

addchild!(x::Union{Term.Progress.ProgressBar,Term.Progress.ProgressJob}; owner=nothing, kwargs...) = addchild!(
    owner, WarmupHMC.Progress(x, (;owner, owns=Set(), labels=Dict(), kwargs...))
)
addchild!(::Nothing, x::WarmupHMC.Progress) = x
addchild!(owner::WarmupHMC.Progress, x::WarmupHMC.Progress) = (push!(owns(owner), x); x)
always_true(;kwargs...) = true
childfilter(owner::WarmupHMC.Progress) = get(root(owner).info, :filter, always_true)
acceptschild(owner::WarmupHMC.Progress; kwargs...) = level(owner) < max_level(owner) && childfilter(owner)(;kwargs...)

WarmupHMC.Progress(::Type{Term.Progress.ProgressBar}; kwargs...) = WarmupHMC.Progress(Val(Term.Progress.ProgressBar), kwargs)
WarmupHMC.initialize_progress!(p::WarmupHMC.Progress{Val{Term.Progress.ProgressBar}}) = WarmupHMC.initialize_progress!(
    Term.Progress.ProgressBar;
    p.info...
)
WarmupHMC.initialize_progress!(p::WarmupHMC.Progress{Val{Term.Progress.ProgressBar}}, N;) = WarmupHMC.initialize_progress!(
    Term.Progress.ProgressBar, N;
    p.info...
)
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}, N; description="Running...", kwargs...) = WarmupHMC.initialize_progress!(
    WarmupHMC.initialize_progress!(Term.Progress.ProgressBar; kwargs...); 
    N, description, propagates=true
)
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}; width=120, kwargs...) = begin 
    bar = Term.Progress.ProgressBar(;width, columns=[DescriptionColumn, CompletedColumn, SeparatorColumn, ProgressColumn])
    Term.Progress.start!(bar)
    thread = Threads.@spawn renderloop(bar)
    addchild!(bar; thread, kwargs...)
end
WarmupHMC.initialize_progress!(owner::WarmupHMC.Progress, N; kwargs...) = WarmupHMC.initialize_progress!(owner; N, kwargs...)
WarmupHMC.initialize_progress!(owner::WarmupHMC.Progress; key=nothing, value="", propagates=false, kwargs...) = if acceptschild(owner; key, value, kwargs...)
    job = Term.Progress.addjob!(parent(root(owner)); id=Base.UUID(rand(UInt128)), kwargs...)
    isnothing(get(kwargs, :N, nothing)) && splice!(job.columns, 2:length(job.columns), (StringColumn(job, value), ))
    addchild!(job; owner, propagates)
end

WarmupHMC.update_progress!(job::Term.Progress.ProgressJob, i::Integer) = Term.Progress.update!(job; i=i-job.i)
WarmupHMC.update_progress!(::Term.Progress.ProgressJob, ::Nothing) = nothing
WarmupHMC.update_progress!(job::Term.Progress.ProgressJob, value) = job.columns[end].msg[] = string(value)
WarmupHMC.update_progress!(job::ProgressJob, i=parent(job).i+1; kwargs...) = begin 
    WarmupHMC.update_progress!(parent(job), i)
    for (key, value) in pairs(kwargs)
        sjob = get!(labels(job), key) do
            skey = rpad(string(key), maximum(length ∘ string, keys(kwargs)))
            WarmupHMC.initialize_progress!(job; key, description="$skey: ")
        end
        WarmupHMC.update_progress!(sjob, value)
    end
    WarmupHMC.update_progress!(owner(job), nothing)
end
WarmupHMC.update_progress!(bar::ProgressBar, ::Nothing) = yield()
WarmupHMC.finalize_progress!(job::ProgressJob) = begin 
    Term.Progress.stop!(parent(job))
    pop!(owns(owner(job)), job)
    propagates(job) && WarmupHMC.finalize_progress!(owner(job))
end
WarmupHMC.finalize_progress!(bar::ProgressBar) = (Term.Progress.stop!(parent(bar)); yield())

end