using Revise
includet("common.jl")
using Serialization

nc_draws = Serialization.deserialize("examples/cache/unconstrained_chains_DynamicDistribution{:ReparametrizedDistribution}_6759533281334098494")

# Plotter(
#     heatmap!,
#     (nc_draws.thinned.thinned.thinned.centeredness_losses, )
# ).markdown