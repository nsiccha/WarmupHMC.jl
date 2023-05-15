data {
    int<lower=1> n_opponents;
    int<lower=1> n_sets[n_opponents];
    int<lower=0, upper=n_sets> n_sets_won[n_opponents];
    int<lower=1> n_serves[n_opponents];
    int<lower=0, upper=n_serves> n_serves_won[n_opponents];
}


parameters {
    real mu;                                                // population mean of success log-odds
    real<lower=0> sigma;                                    // population sd of success log-odds
    vector[n_opponents] alpha;                              // success log-odds
}

model {
    mu ~ normal(0, 1);                                      // hyperprior
    sigma ~ normal(0, 1);                                   // hyperprior
    alpha ~ normal(mu, sigma);                              // prior (hierarchical)
    n_serves_won ~ binomial_logit(n_serves, alpha);         // likelihood
}