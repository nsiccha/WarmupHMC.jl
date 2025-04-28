module TermExt
import Term, WarmupHMC

ProgressBar{I} = WarmupHMC.Progress{Term.Progress.ProgressBar, I}
ProgressJob{I} = WarmupHMC.Progress{Term.Progress.ProgressJob, I}

import Term.Progress: AbstractColumn, DescriptionColumn, CompletedColumn, SeparatorColumn, ProgressColumn
struct MaybeProgressColumn <: AbstractColumn
    parent::ProgressColumn
    measure::Term.Progress.Measure
    msg::Ref{String}
    MaybeProgressColumn(job::Term.Progress.ProgressJob) = MaybeProgressColumn(ProgressColumn(job))
    MaybeProgressColumn(parent::ProgressColumn) = new(parent, parent.measure, Ref(""))
end 
Term.Progress.update!(col::MaybeProgressColumn, args...) = isnothing(col.parent.job.N) ? col.msg[] : Term.Progress.update!(col.parent, args...)

function renderloop(pbar)
    while pbar.running
        pbar.paused || Term.Progress.render(pbar)
        sleep(pbar.Δt)
    end
end


WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}, N) = begin
    WarmupHMC.initialize_progress!(WarmupHMC.initialize_progress!(Term.Progress.ProgressBar); N)
end
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}; kwargs...) = begin 
    bar = Term.Progress.ProgressBar(width=120 ; columns=[DescriptionColumn, CompletedColumn, SeparatorColumn, ProgressColumn])
    Term.Progress.start!(bar)
    WarmupHMC.Progress(bar, Threads.@spawn renderloop(bar))
end
WarmupHMC.initialize_progress!(bar::ProgressBar, N) = WarmupHMC.initialize_progress!(bar; N)
WarmupHMC.initialize_progress!(bar::ProgressBar; kwargs...) = begin
    WarmupHMC.Progress(Term.Progress.addjob!(parent(bar); id=Base.UUID(rand(UInt128)), kwargs...), Dict{Symbol,Any}(:top=>bar))
end
WarmupHMC.initialize_progress!(job::ProgressJob; kwargs...) = begin
    rv = WarmupHMC.initialize_progress!(job.info[:top]; kwargs...)
    if isnothing(get(kwargs, :N, nothing))
        splice!(parent(rv).columns, 2:length(parent(rv).columns), (MaybeProgressColumn(parent(rv)),))
    end
    rv
end 
WarmupHMC.update_progress!(job::ProgressJob, i, info=(;)) = begin 
    WarmupHMC.update_progress!(parent(job), i)
    bar = job.info[:top]
    if length(info) > 0
        skeys = get!(job.info, :skeys) do 
            max_length = maximum(length ∘ string, keys(info))
            Dict([
                key=>rpad(string(key), max_length) for key in keys(info)
            ])
        end
        for (key, value) in pairs(info)
            skey = skeys[key]
            sjob = get!(job.info, key) do
                WarmupHMC.initialize_progress!(job; description="$skey: ")
            end
            parent(sjob).columns[end].msg[] = string(value)
        end
    end
    WarmupHMC.update_progress!(bar)
    job
end
WarmupHMC.update_progress!(job::ProgressJob) = begin 
    WarmupHMC.update_progress!(parent(job))
    WarmupHMC.update_progress!(job.info[:top])
end
WarmupHMC.update_progress!(job::Term.Progress.ProgressJob, i=job.i+1) = begin 
    Term.Progress.update!(job; i=i-job.i)
    job
end
WarmupHMC.update_progress!(::ProgressBar) = yield()
WarmupHMC.finalize_progress!(job::ProgressJob, args...) = begin 
    Term.Progress.stop!(parent(job))
    WarmupHMC.finalize_progress!(job.info[:top], args...)
end
WarmupHMC.finalize_progress!(bar::ProgressBar, args...) = nothing
WarmupHMC.finalize_progress!(bar::ProgressBar, ::Type, args...) = Term.Progress.stop!(parent(bar))

end