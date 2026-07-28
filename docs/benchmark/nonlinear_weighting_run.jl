# Re-measures the nonlinear online trajectory-weighting study.
#
#   julia --project=docs/benchmark docs/benchmark/nonlinear_weighting_run.jl
#
# Writes `results/nonlinear_weighting/rows.json` — ROWS ONLY, plus the config
# that produced them. Every summary is derived from those rows by
# `nonlinear_weighting.jl`; nothing derived is stored here. Read the header of
# that file for why.
#
# Configurable through the environment, so one committed driver serves both the
# CI smoke run and the full study:
#
#   WHMC_NW_OUT       output directory  (default results/nonlinear_weighting)
#   WHMC_NW_SEEDS     e.g. "1:20" or "1,2,3"            (default 1:20)
#   WHMC_NW_TARGETS   comma-separated keys from TARGET_SPECS (default all three)
#   WHMC_NW_DRAWS     override every target's draw count (default per-target)
#   WHMC_NW_EVIDENCE  comma-separated nonlinear_evidence values
#   WHMC_NW_WEIGHTING comma-separated nonlinear_trajectory_weighting values
#
# THIS DRIVER ASSERTS NOTHING ABOUT THE OUTCOME. It exits non-zero only when a
# run actually fails, never because a policy came out behind — a benchmark that
# fails CI on an unfavourable result teaches everyone to stop looking at it, and
# these rows are noisy by nature. CI's job here is to prove the driver, the
# schema and the artifact upload still work; the numbers worth quoting come
# from a deliberate quiet-host run with its SHA recorded. See README.md.
include(joinpath(@__DIR__, "common.jl"))

using WarmupHMC: adaptive_warmup_mcmc
import JSON

# The two cross-target checks are ANALYTIC targets, not posteriordb Stan
# problems, and they live in the verification suite. Included here rather than
# copied: a second copy of `EightSchools` would be free to drift away from the
# one the tests actually exercise, and the first symptom of that drift would be
# a benchmark disagreeing with a test for no visible reason. Namespaced in a
# module because `common.jl` defines its own (larger) `Funnel`, and Julia will
# not let one module hold two structs of that name.
#
# `web/src/test/` is owned by `WarmupHMC:reparam-verify`. This is a read-only
# include across that boundary, nothing more — no file there is modified.
module StudyTargets
include(joinpath(@__DIR__, "..", "..", "web", "src", "test", "targets.jl"))
end

# The two policy axes. `nonlinear_evidence` selects which leaves contribute
# evidence; `nonlinear_trajectory_weighting` selects how that evidence is
# scaled. Both are validated by the sampler, which names the accepted values on
# a bad one — so a renamed policy fails loudly here rather than silently
# measuring the default four times.
const EVIDENCE  = split(get(ENV, "WHMC_NW_EVIDENCE",
                            "all_good_leaves,nuts_weighted"), ',')
const WEIGHTING = split(get(ENV, "WHMC_NW_WEIGHTING", "unit,stepsize"), ',')

# Draw counts are per-target and deliberately unequal: radon is dim 88 and the
# expensive one, the two dim-10 targets are the cheap cross-target check.
const TARGET_SPECS = Dict(
    "radon_mn-radon_partially_pooled_centered" => (; pdb = "radon_mn-radon_partially_pooled_centered", n_draws = 500),
    "eight_schools" => (; pdb = nothing, n_draws = 1000),
    "funnel"        => (; pdb = nothing, n_draws = 1000),
)

parse_seeds(s) = occursin(':', s) ?
    collect(range(parse.(Int, split(s, ':'))...)) : parse.(Int, split(s, ','))

const SEEDS = parse_seeds(get(ENV, "WHMC_NW_SEEDS", "1:20"))
const TARGET_KEYS = split(get(ENV, "WHMC_NW_TARGETS",
    "radon_mn-radon_partially_pooled_centered,funnel,eight_schools"), ',')
const DRAWS_OVERRIDE = haskey(ENV, "WHMC_NW_DRAWS") ?
    parse(Int, ENV["WHMC_NW_DRAWS"]) : nothing
const OUT = env_dir("WHMC_NW_OUT",
                    joinpath(@__DIR__, "results", "nonlinear_weighting"))

for k in TARGET_KEYS
    haskey(TARGET_SPECS, k) || error("unknown target `$k`; known: " *
                                     join(sort(collect(keys(TARGET_SPECS))), ", "))
end

"""
    build(key) -> (problem, spec, ad_backend)

The problem and its reparametrization spec, exactly as the study measured them.

Each construction is pinned to what produced the checked-in rows, not to what
the rest of this benchmark happens to use. The funnel is the clearest example:
`docs/benchmark/` elsewhere uses `Funnel(10)`, the study used K = 5, and a
driver quietly substituting the other would be a different experiment wearing
the same name.
"""
function build(key)
    if key == "funnel"
        # K = 5, dimension 6: v ~ Normal(0, 3) and five theta_i of scale
        # exp(v/2). Source AND target both centered — unlike `funnel_spec` in
        # common.jl, which parametrizes the target.
        f = StudyTargets.Funnel(5)
        spec = IndexedReparametrization([
            (i + 1) => Reparametrization(PartiallyCentered(1.0),
                                         PartiallyCentered(1.0), 0.0,
                                         x -> x[1] / 2) for i in 1:5])
        return f, spec, AutoForwardDiff()
    elseif key == "eight_schools"
        # The ANALYTIC eight schools, not the posteriordb Stan problem of the
        # same name; the spec is the shipped posteriordb one applied to it.
        return StudyTargets.EightSchools(true),
               reparametrization("eight_schools-eight_schools_centered", 10, nothing),
               AutoForwardDiff()
    end
    spec_info = TARGET_SPECS[key]
    problem, dim, jdata = stan_problem(spec_info.pdb)
    # `AutoForwardDiff()`, NOT the harness's `AD_BACKEND` — the backend is
    # ENCODED here rather than inherited, so `WHMC_BENCH_AD` does not reach this
    # driver. Every target in this study is pinned to the backend its measured
    # construction used; a study whose backend comes from the environment is not
    # a study, because the environment is not in the artifact.
    #
    # It happens not to matter for THIS target, and the exception is worth
    # knowing rather than relying on: measured here, radon under
    # `AutoEnzyme(; function_annotation = Enzyme.Const)` reproduces the rows of
    # record BIT-FOR-BIT — seeds 1-2, all four policy arms, all of ess_min,
    # ess_median, grad_evals, n_divergent and final_c. That is specific to a Stan
    # target: BridgeStan supplies the log-density gradient analytically and the
    # AD backend differentiates only the wrapper's `ljac_(x_) + dot(g_y, y_)`,
    # which the two backends agree on to the last bit here. Do NOT generalise it
    # — on the ANALYTIC targets above, where AD computes the whole gradient, the
    # backend changes the trajectory and the run (measured on eight_schools:
    # 613.73 min-ESS / 13045 gradients under Enzyme against 606.08 / 13061 under
    # ForwardDiff, same seed).
    problem, native_spec(spec_info.pdb, dim, jdata), AutoForwardDiff()
end

# One row per (target, seed, evidence, weighting). `order_index` records WHERE
# in the seed's execution order this configuration ran, because on a shared host
# a configuration that always runs first is not comparable with one that always
# runs last — the order is rotated per seed and the index is kept so that a
# reader can check the rotation actually happened instead of trusting it.
rows = Dict{String,Any}[]
configs = [(e, w) for e in EVIDENCE for w in WEIGHTING]

for key in TARGET_KEYS
    spec_info = TARGET_SPECS[key]
    n_draws = something(DRAWS_OVERRIDE, spec_info.n_draws)
    @info "target" key n_draws n_configs = length(configs) n_seeds = length(SEEDS)
    for seed in SEEDS
        # Rotate by seed, so no configuration keeps a systematically warm or
        # cold slot across the study.
        rotated = circshift(configs, seed - 1)
        for (order_index, (evidence, weighting)) in enumerate(rotated)
            problem, spec, ad = build(key)  # rebuilt per run: the spec is mutated
            lpdf = ReparametrizedProblem(spec, problem, ad)
            res = adaptive_warmup_mcmc(Xoshiro(seed), lpdf; n_draws = n_draws,
                                       nonlinear_evidence = Symbol(evidence),
                                       nonlinear_trajectory_weighting = Symbol(weighting))
            draws = Matrix{Float64}(res.posterior_position)
            ess = ess_per_coordinate(draws)
            push!(rows, Dict{String,Any}(
                "target" => key, "seed" => seed,
                "leaf_policy" => String(evidence), "metric_policy" => String(weighting),
                "order_index" => order_index, "n_draws" => n_draws,
                "grad_evals" => Int(res.total_evaluation_counter),
                "ess_min" => nanmin(ess), "ess_median" => nanmed(ess),
                "n_divergent" => Int(res.n_divergent_samples),
                "final_c" => source_cs(spec)))
        end
    end
end

# Sampler settings the runs did NOT pass, recorded as their RESOLVED VALUES
# rather than as the word "defaults". "Defaults" names a moving target: it
# resolves against whatever the sampler shipped on the day, so a config saying
# "defaults" describes a different experiment after any change to them, and
# nothing in the file marks the day it moved. These are the values in force at
# the recorded SHA. If the driver ever starts passing one explicitly, move it
# out of this block and into the call.
const RESOLVED_DEFAULTS = Dict{String,Any}(
    "n_evaluations" => 1000, "stepsize_adaptation_limit" => 50,
    "recording_target" => 1000, "target_acceptance_rate" => 0.8,
    "max_tree_depth" => 10, "nonlinear_adapt" => true,
    "init" => "sampler default (Pathfinder from Uniform(-2, 2))")

# How each target was built, in the same file as the numbers. A target named
# only by a string is not reproducible: `funnel` and `eight_schools` here are
# ANALYTIC targets from web/src/test/targets.jl, and both differ from the
# same-named things elsewhere in this benchmark (`Funnel(10)`) and in
# posteriordb (the eight-schools Stan model). Guessing from the name cost a
# full round of mismatched rows before this block existed.
target_provenance(k) =
    k == "funnel" ? Dict("constructor" => "StudyTargets.Funnel(5)",
                         "source" => "web/src/test/targets.jl",
                         "dimension" => 6, "n_pairs" => 5,
                         "spec" => "IndexedReparametrization, (i+1) => " *
                                   "Reparametrization(PartiallyCentered(1.0), " *
                                   "PartiallyCentered(1.0), 0.0, x -> x[1]/2), i in 1:5",
                         "ad_backend" => "AutoForwardDiff") :
    k == "eight_schools" ? Dict("constructor" => "StudyTargets.EightSchools(true)",
                         "source" => "web/src/test/targets.jl",
                         "dimension" => 10, "n_pairs" => 8,
                         "spec" => "reparametrization(\"eight_schools-" *
                                   "eight_schools_centered\", 10, nothing)",
                         "ad_backend" => "AutoForwardDiff") :
    Dict("constructor" => "stan_problem(\"$k\") — posteriordb via BridgeStan",
         "source" => "posteriordb",
         "spec" => "native_spec(\"$k\", dim, jdata)",
         "ad_backend" => "AutoForwardDiff",
         "ad_backend_cross_check" =>
             "AutoEnzyme(; function_annotation = Enzyme.Const) reproduces these " *
             "rows bit-for-bit (seeds 1-2, all four arms, every identity " *
             "field); BridgeStan supplies this target's gradient analytically, " *
             "so the backend only differentiates the wrapper term. Not true of " *
             "the analytic targets.")

config = Dict{String,Any}(
    "study" => "nonlinear online trajectory weighting: leaf policy x metric policy",
    "measured_by" => get(ENV, "KB_AGENT_ID", "unknown"),
    "host" => get(ENV, "KB_HOST", "unknown"),
    # `warmuphmc_sha`, `worktree_dirty` and `src_dirty` are merged in below from
    # `git_provenance()` (common.jl) rather than spelled out here, so this file
    # cannot drift into recording the revision without the cleanliness flags.
    "julia" => string(VERSION),
    # READ, not asserted: `common.jl` calls `BLAS.set_num_threads(1)` because a
    # multithreaded BLAS reduces in a nondeterministic order, which changes the
    # trajectory and so the draws at a fixed seed. The host's default here is 4,
    # so this is doing real work — recording the live value means an artifact
    # measured with that line removed says so instead of looking identical.
    "blas_threads" => BLAS.get_num_threads(),
    "seeds" => SEEDS,
    "leaf_policies" => String.(EVIDENCE),
    "metric_policies" => String.(WEIGHTING),
    "resolved_sampler_defaults" => RESOLVED_DEFAULTS,
    "targets" => Dict(k => merge(
                        Dict("n_draws" => something(DRAWS_OVERRIDE,
                                                    TARGET_SPECS[k].n_draws),
                             "n_runs" => count(r -> r["target"] == k, rows)),
                        target_provenance(k))
                      for k in TARGET_KEYS),
    "note" => "Rows only. Every summary is derived by nonlinear_weighting.jl at " *
              "read time; regenerate the rows with nonlinear_weighting_run.jl.")

merge!(config, git_provenance())

mkpath(OUT)
open(joinpath(OUT, "rows.json"), "w") do io
    JSON.print(io, Dict("config" => config, "rows" => rows), 2)
end
@info "wrote" path = joinpath(OUT, "rows.json") n_rows = length(rows)
