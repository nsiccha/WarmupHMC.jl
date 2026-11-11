using JSON, Statistics, Dates, Pkg, WarmupHMC

haskey(ENV, "CI") && error("Documentation fits must be generated outside CI")
include(joinpath(pkgdir(WarmupHMC), "docs/cache.jl"))
include("gaussian.jl")
include("checkpoint.jl")

draws = result.posterior_position
summary = Dict(
    "shape" => collect(size(draws)),
    "coordinates" => [Dict(
        "coordinate" => i, "mean" => mean(draws[i, :]),
        "sd" => std(draws[i, :]), "ess" => result.ess[i],
        "mean_mcse" => std(draws[i, :]) / sqrt(result.ess[i]),
        "q05" => quantile(draws[i, :], .05), "q95" => quantile(draws[i, :], .95),
    ) for i in axes(draws, 1)],
    "diagnostics" => Dict(
        "divergent_draws" => result.n_divergent_samples,
        "sampling_gradient_evaluations" => result.sampling_evaluation_counter,
        "run_gradient_evaluations" => result.total_evaluation_counter,
        "stepsize" => result.stepsize,
        "transformation" => string(result.active_transformation),
    ),
    "checkpoint" => checkpoint_summary,
    "receipt" => Dict(
        "program_sha256" => ManualCache.digest(ManualCache.PROGRAMS),
        "source_sha256" => ManualCache.source_digest(),
        "julia" => string(VERSION), "warmuphmc" => string(pkgversion(WarmupHMC)),
        "base_revision" => readchomp(`git -C $(pkgdir(WarmupHMC)) rev-parse HEAD`),
        "created_utc" => string(now(UTC)),
        "dependencies" => Dict(info.name => string(info.version) for (_, info) in Pkg.dependencies()
                               if info.name in ["DynamicHMC", "LogDensityProblems", "Pathfinder", "MCMCDiagnosticTools"]),
        "command" => "julia --project=docs bench/docs/run.jl",
    ),
)
output = joinpath(pkgdir(WarmupHMC), "docs/data/gaussian.json")
write(output, JSON.json(summary, 2) * "\n")
println("Saved summary and receipt: ", output)
println("Draw matrix: ", size(draws), "; checkpoint: ", checkpoint_summary)
