using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal
using Enzyme
using Random
using StanBlocks
using WarmupHMC

# A reusable BRM formula with one scalar random intercept per subject.
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end

data = (
    subject = repeat([11, 12], inner=3),
    x = collect(range(-1.0, 1.0; length=6)),
    y = zeros(6),
)

# Compile the BRM model, then let BRM discover the coordinates WarmupHMC
# should adapt. The default SBBRMI emission is the non-centred c=0 endpoint.
brm = SBBRMI(builder(data); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(brm.model)
unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
blocks = adaptive_centering_blocks(brm, unc_names)

block_summary = [
    (; binding=block.ranef.binding,
       cells=size(block.effects),
       target_c=block.target_c)
    for block in blocks
]
@show block_summary

# BRM constructs the custom fixed-frame CandidateScoringPlan; no raw coordinate
# indices or Reparametrization objects need to be assembled by the caller.
backend = AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const,
)
adaptive_problem = adaptive_centering_problem(brm, problem, backend)
result = adaptive_warmup_mcmc(
    Xoshiro(20260728), adaptive_problem;
    n_draws=40,
    progress=nothing,
)

# A single-chain run fits the source centerings in place. Returned draws have
# already been transported back to the compiled BRM model's own coordinates.
learned_centerings = [
    value.source.c
    for (_, value) in WarmupHMC.reparametrizer(adaptive_problem).pairs
]
@show size(result.posterior_position) learned_centerings
