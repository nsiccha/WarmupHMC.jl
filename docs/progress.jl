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

problem = pdb_problem("radon_all-radon_variable_intercept_slope_noncentered")
WarmupHMC.adaptive_warmup_mcmc(Xoshiro.(1:4), problem; n_draws=100, progress=Term.Progress.ProgressBar)
nothing