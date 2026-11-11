using WarmupHMC, Random

checkpoint_summary = mktempdir() do directory
    first_run = adaptive_warmup_mcmc(Xoshiro(83), Gaussian();
        n_draws=200, checkpoint_dir=directory, progress=nothing)
    resumed = adaptive_warmup_mcmc(Xoshiro(999), Gaussian();
        n_draws=400, checkpoint_dir=directory, resume=true, progress=nothing)
    n_before = size(first_run.posterior_position, 2)
    (; before=n_before, after=size(resumed.posterior_position, 2),
       prefix_preserved=resumed.posterior_position[:, 1:n_before] ==
                        first_run.posterior_position)
end
