module TermExt
import Term, WarmupHMC

WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}, N) = begin
    bar = Term.Progress.ProgressBar(; columns=:detailed)
    job = Term.Progress.addjob!(bar; N)
    Term.Progress.start!(bar)
    WarmupHMC.Progress(bar, job)
end
WarmupHMC.update_progress!(p::WarmupHMC.Progress{Term.Progress.ProgressBar}, i, info=(;)) = begin 
    # i < p.info.i && println("DECREASING $(p.info.i=>i)")
    job = p.info
    Term.Progress.update!(job; i=i-job.i)
    Term.Progress.render(parent(p))
    p
end
WarmupHMC.finalize_progress!(p::WarmupHMC.Progress{Term.Progress.ProgressBar}, i, info=(;)) = begin 
    WarmupHMC.update_progress!(p, i, info)
    Term.Progress.stop!(parent(p))
    p
end
WarmupHMC.initialize_progress!(::Type{Term.Progress.ProgressBar}; n_chains, n_draws) = begin 
    bar = Term.Progress.ProgressBar(; columns=:detailed)
    rv = map(i->WarmupHMC.Progress(Term.Progress.addjob!(bar; N=n_draws), bar), 1:n_chains)
    Term.Progress.start!(bar)
    Threads.@spawn Term.Progress._startrenderloop(bar)
    rv
end

WarmupHMC.initialize_progress!(p::WarmupHMC.Progress{Term.Progress.ProgressJob}, N) = begin
    parent(p).N = N
    p
end
WarmupHMC.update_progress!(p::WarmupHMC.Progress{Term.Progress.ProgressJob}, i, info=(;)) = begin 
    job = parent(p)
    Term.Progress.update!(job; i=i-job.i)
    p
end
WarmupHMC.finalize_progress!(p::WarmupHMC.Progress{Term.Progress.ProgressJob}, i, info=(;)) = begin 
    WarmupHMC.update_progress!(p, i, info)
    p
end
WarmupHMC.finalize_progress!(p::Vector{<:WarmupHMC.Progress{Term.Progress.ProgressJob}}) = begin 
    Term.Progress.stop!(first(p).info)
end

end