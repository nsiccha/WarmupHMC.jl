using WarmupHMC, PosteriorDB, StanLogDensityProblems, LogDensityProblems, Logging, Random, Term


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
for progress in (Term.Progress.ProgressBar, )
    @time WarmupHMC.adaptive_warmup_mcmc(Xoshiro.(1:4), problem; n_draws=10_000, progress)
    @time WarmupHMC.adaptive_warmup_mcmc(Xoshiro.(1:8), problem; n_draws=10_000, progress)
    @time WarmupHMC.adaptive_warmup_mcmc(Xoshiro.(1:16), problem; n_draws=10_000, progress)
    @time WarmupHMC.adaptive_warmup_mcmc(Xoshiro.(1:32), problem; n_draws=10_000, progress)
end
end
