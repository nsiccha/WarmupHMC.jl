using WarmupHMC, PosteriorDB, StanLogDensityProblems, LogDensityProblems, Random, Term, Treebars

const pdb = PosteriorDB.database()

pdb_problem(posterior_name) = begin
    posterior_name = posterior_name |> strip |> String
    posterior = PosteriorDB.posterior(pdb, posterior_name)
    WarmupHMC.NamedPosterior(StanProblem(
        PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan")), 
        PosteriorDB.load(PosteriorDB.dataset(posterior), String);
        nan_on_error=true,
        make_args=["STAN_THREADS=TRUE"],
        warn=false
    ), posterior_name);
end

begin
problem = pdb_problem("eight_schools-eight_schools_noncentered")

struct MultiProgress{P<:Tuple}
    parents::P
    MultiProgress(args...) = new{typeof(args)}(args)
end

Treebars.initialize_progress!(p::MultiProgress, args...; kwargs...) = MultiProgress(
    map(parent->Treebars.initialize_progress!(parent, args...; kwargs...), p.parents)...
)
Treebars.update_progress!(p::MultiProgress, args...; kwargs...) = (map(
    parent->Treebars.update_progress!(parent, args...; kwargs...), p.parents
); yield())
Treebars.finalize_progress!(p::MultiProgress, args...; kwargs...) = map(
    parent->Treebars.finalize_progress!(parent, args...; kwargs...), p.parents
)



struct RemoteProgress
    lock::ReentrantLock
    info::Ref{NamedTuple}
    RemoteProgress(info=(;)) = new(ReentrantLock(), Ref{NamedTuple}(info))
end
info(p::RemoteProgress) = lock(p.lock) do 
    p.info[]
end
Treebars.initialize_progress!(p::RemoteProgress, N::Integer; kwargs...) = RemoteProgress(merge((;N, i=0, kwargs...)))
Treebars.update_progress!(p::RemoteProgress, args...; kwargs...) = nothing
Treebars.update_progress!(p::RemoteProgress, msg::AbstractString; kwargs...) = lock(p.lock) do
    p.info[] = merge(p.info[], (;msg))
end
Treebars.update_progress!(p::RemoteProgress, i::Integer; kwargs...) = lock(p.lock) do 
    p.info[] = merge(p.info[], (;i))
end
Treebars.fail_progress!(p::RemoteProgress, args...; kwargs...) = lock(p.lock) do 
    p.info[] = merge(p.info[], (;failed=true))
end
Treebars.finalize_progress!(p::RemoteProgress, args...; kwargs...) = lock(p.lock) do 
    p.info[] = merge(p.info[], (;done=true))
end
Base.show(io::IO, p::Treebars.ProgressNode{<:RemoteProgress}) = begin 
    info_ = info(p.impl)
    print(io, info_.description, ":\n")
    if haskey(info_, :msg)
        print(io, info_.msg, "\n")
    elseif haskey(info_, :N)
        (;i, N) = info_
        print(io, i, " of ", N, " done.\n")
    end
    for child in p.children
        print(io, child)
    end
end

using DynamicObjects

@dynamicstruct struct Dummy
    problem
    status[seed, n_draws] = Treebars.ProgressNode(RemoteProgress((;description="Fitting $seed with $n_draws")))
    fit[seed, n_draws] = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem; progress=status[seed, n_draws], n_draws)
end

d = Dummy(problem; cache_type=:parallel)


ip = d.fit

myfetch(x) = if isa(x, Task)
    @info d.status[1, 1_000]
else
    @info d.status[1, 1_000]
    x
end

while true
    status = getindex(d.fit, 1, 1_000; fetch=myfetch)
    if isa(status, NamedTuple)
        @info "Done!"
        break
    end 
    sleep(.01)
end

error()

mp = Treebars.ProgressNode(MultiProgress(:term, RemoteProgress()))
WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), problem; n_draws=1_000, progress=(Treebars, mp))
error()
end
begin

begin
@time rv = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), problem; progress=Term.ProgressBar, n_draws=10_000)
WarmupHMC.MCMCDiagnosticTools.ess(reshape(rv.posterior_position', (:, 1, size(rv.posterior_position, 1)))) |> extrema |> display
end

using DynamicHMC
begin
@time results = mcmc_with_warmup(Xoshiro(1), problem, 10_000; reporter = ProgressMeterReport())
WarmupHMC.MCMCDiagnosticTools.ess(reshape(results.posterior_matrix', (:, 1, size(results.posterior_matrix, 1)))) |> extrema |> display
end
end