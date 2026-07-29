# Does nonlinear reparametrization help on REAL catalogue posteriors?
#
#   julia --startup-file=no --project=docs/benchmark/brm \
#     docs/benchmark/run_brm_catalogue_benchmark.jl
#
# Optional environment knobs:
#   BRMB_SEEDS=12     chain seeds per (spec, arm, flag)
#   BRMB_DRAWS=500    retained draws per chain
#   BRMB_SPECS=       comma-separated spec keys; default all
#   BRMB_OUT=docs/benchmark/results/brm_catalogue/rows.json
#   BRM_DATA_CACHE=docs/benchmark/brm/data
#
# WHAT THIS MEASURES, AND THE TRAP IT IS BUILT AROUND
#
# `nonlinear_adapt=true` does NOTHING on a target that carries no
# reparametrizer. `WarmupHMC.reparametrizer(::Any)` returns an empty
# `IndexedReparametrization` (src/Reparametrizations.jl), and
# `find_reparametrization!` short-circuits on `isempty(ir.pairs)`. There is no
# automatic detection of hierarchical structure.
#
# So varying the flag over a bare BridgeStan problem produces a complete,
# well-formed table of numbers that measures nothing at all. This benchmark runs
# exactly that configuration ON PURPOSE, as two of its three arms: the
# `noncentered` and `centered` rows are CONTROLS, and their flag-on/flag-off
# pairs must come back bit-identical. If they ever differ, the wiring changed
# and the third arm's result is no longer interpretable. A knob wired to nothing
# still reports numbers; that is the whole reason the controls are here.
#
# The third arm wraps the model through `BRM.adaptive_centering_problem`, which
# builds a `PartiallyCentered` reparametrization with per-(block, term, group)
# centeredness. That one has pairs, so the flag reaches something.
#
# WHY ESS IS TAKEN IN THE CONSTRAINED SPACE, OVER SHARED NAMES ONLY
#
# The centered and non-centered models are different Stan programs. They declare
# different constrained parameter sets AND completely different unconstrained
# coordinates — the non-centered one samples `z_flat` innovations, the centered
# one samples the random effects themselves. A min-ESS over each model's own
# coordinates compares different quantities and silently favours whichever
# model happens to declare fewer awkward ones. Everything below is reduced over
# the INTERSECTION of the two models' constrained names. This is the same trap
# `common.jl` documents for the posteriordb siblings.
#
# WHY CONSTANT COORDINATES ARE DROPPED, AND COUNTED
#
# A correlation Cholesky has structurally fixed entries (`L[1,1] = 1`,
# `L[1,2] = 0`). `MCMCDiagnosticTools.ess` returns NaN for a constant chain and
# `minimum` propagates it, so a single structurally-constant coordinate turns
# the whole run's min-ESS into NaN — which reads as "the run failed" when the
# run was fine. They are filtered, and `n_constant` records how many were
# dropped, so "constant" can never be mistaken for "broken".

# WHY THE BLAS PIN IS LOAD-BEARING
#
# It is not hygiene. BLAS thread count CHANGES THE SAMPLED CHAIN on at least one
# of these targets. Measured: the `centered` sleepstudy arm run at the box's
# default 8 threads returns different per-seed gradient counts and a different
# min-ESS than the same seed at 1 thread; pinned to 1 it reproduces exactly
# (seeds 1-3 give 10662 / 16062 / 16838 both times). Unpinned, this benchmark
# would not be reproducible at all, and two runs of it would not be comparable.
#
# It shows on the centered arm and not on `noncentered` or `adaptive_centering`,
# which are bit-identical across both settings. That asymmetry is NOT explained
# — the two parametrizations are different generated Stan programs and so do not
# perform the same linear algebra, but which call is thread-order-sensitive has
# not been established. `blas_threads` is recorded in the artifact config so any
# future comparison is against a known setting rather than an assumed one.
using LinearAlgebra
BLAS.set_num_threads(1)

using Dates, Distributions, JSON, LogDensityProblems, MCMCDiagnosticTools
using Random, SHA, Statistics
using BayesianRegressionModels, StanBlocks, WarmupHMC
using Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

include(joinpath(@__DIR__, "brm", "models.jl"))

const AD_BACKEND = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                                function_annotation = Enzyme.Const)

const N_SEEDS = parse(Int, get(ENV, "BRMB_SEEDS", "12"))
const N_DRAWS = parse(Int, get(ENV, "BRMB_DRAWS", "500"))
const OUT = get(ENV, "BRMB_OUT",
                joinpath(@__DIR__, "results", "brm_catalogue", "rows.json"))

function selected()
    specs = haskey(ENV, "BRMB_SPECS") ?
        filter(s -> s.key in split(ENV["BRMB_SPECS"], ","), SPECS) : copy(SPECS)
    # Cheapest-first; see RUN_ORDER in brm/models.jl for the measured costs. A
    # spec missing from RUN_ORDER sorts last rather than throwing, so adding a
    # model does not require touching two places to run it.
    sort(specs; by = s -> something(findfirst(==(s.key), RUN_ORDER), length(RUN_ORDER) + 1))
end

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

function gitsha(dir)
    try
        readchomp(`git -C $dir rev-parse HEAD`)
    catch
        "unavailable"
    end
end

function gitdirty(dir, paths...)
    try
        !isempty(readchomp(`git -C $dir status --porcelain -- $(collect(paths))`))
    catch
        true
    end
end

pkgdir_of(m) = dirname(dirname(pathof(m)))

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

function ess_vec(draws::AbstractMatrix)
    d, n = size(draws)
    (n < 10 || !all(isfinite, draws)) && return Float64[]
    collect(MCMCDiagnosticTools.ess(reshape(permutedims(draws), (n, 1, d))))
end

"""Minimum over the finite entries, plus how many were not finite."""
function finite_min(v)
    w = filter(isfinite, v)
    (isempty(w) ? NaN : minimum(w), length(v) - length(w))
end

"""
Constrain `draws` through BridgeStan and reduce min-ESS over `shared` names only.
Returns `(min_ess, n_draws_constrained, n_constant)`.
"""
function shared_constrained_ess(model, names, shared, draws)
    keep = [i for (i, nm) in enumerate(names) if nm in shared]
    out = Matrix{Float64}(undef, length(keep), size(draws, 2))
    ok = falses(size(draws, 2))
    for j in axes(draws, 2)
        try
            v = BS.param_constrain(model, Vector{Float64}(draws[:, j]); include_tp = true)
            out[:, j] = v[keep]
            ok[j] = all(isfinite, @view out[:, j])
        catch
            ok[j] = false
        end
    end
    m, ndrop = finite_min(ess_vec(out[:, ok]))
    (m, count(ok), ndrop)
end

function run_arm(; spec_key, arm, problem, model, names, shared, adapt, seed)
    rng = Xoshiro(seed)
    try
        wall = @elapsed res = adaptive_warmup_mcmc(rng, problem; n_draws = N_DRAWS,
                                                   nonlinear_adapt = adapt,
                                                   progress = nothing)
        draws = Matrix{Float64}(res.posterior_position)
        ess_unc, _ = finite_min(ess_vec(draws))
        ess_con, n_ok, n_const = shared_constrained_ess(model, names, shared, draws)
        grad = Int(res.total_evaluation_counter)
        (; spec = spec_key, arm, nonlinear_adapt = adapt, seed, ok = true,
           wall_s = wall, grad_evals = grad,
           ess_min_unconstrained = ess_unc, ess_min_shared_constrained = ess_con,
           ess_min_per_grad = ess_con / max(grad, 1),
           n_draws_constrained = n_ok, n_constant = n_const,
           n_divergent = Int(res.n_divergent_samples), error = "")
    catch err
        (; spec = spec_key, arm, nonlinear_adapt = adapt, seed, ok = false,
           wall_s = NaN, grad_evals = 0,
           ess_min_unconstrained = NaN, ess_min_shared_constrained = NaN,
           ess_min_per_grad = NaN, n_draws_constrained = 0, n_constant = 0,
           n_divergent = 0, error = first(sprint(showerror, err), 300))
    end
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

sanitize(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
sanitize(x::AbstractDict) = Dict(k => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]

const ROWS = Any[]
const MODEL_META = Any[]

function flush_out(config)
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        JSON.print(io, Dict("config" => sanitize(config),
                            "models" => sanitize(MODEL_META),
                            "rows" => [sanitize(Dict(string(k) => v for (k, v) in pairs(r)))
                                       for r in ROWS]), 2)
    end
end

function main()
    specs = selected()
    whmc_dir = pkgdir_of(WarmupHMC)
    config = Dict(
        "host" => get(ENV, "KB_HOST", gethostname()),
        "julia" => string(VERSION),
        "blas_threads" => BLAS.get_num_threads(),
        "sampler" => "adaptive_warmup_mcmc",
        "runner" => "docs/benchmark/run_brm_catalogue_benchmark.jl",
        "reproduction" => "BRMB_SEEDS=$(N_SEEDS) BRMB_DRAWS=$(N_DRAWS) julia " *
                          "--startup-file=no --project=docs/benchmark/brm " *
                          "docs/benchmark/run_brm_catalogue_benchmark.jl",
        "warmuphmc_sha" => gitsha(whmc_dir),
        "src_dirty" => gitdirty(whmc_dir, "src"),
        "worktree_dirty" => gitdirty(whmc_dir),
        # The two packages that make this benchmark possible are UNREGISTERED.
        # Recording their SHAs is the only pin available; a reader cannot resolve
        # them from a registry, and this file says so rather than implying a
        # `Pkg.instantiate` would reproduce the run.
        "brm_sha" => gitsha(pkgdir_of(BayesianRegressionModels)),
        "stanblocks_sha" => gitsha(pkgdir_of(StanBlocks)),
        "unregistered_dependencies" =>
            "BayesianRegressionModels and StanBlocks are not in any registry. " *
            "Regenerating this artifact requires both at the SHAs above, added " *
            "with Pkg.develop. The docs BUILD does not need them: it reads this " *
            "checked-in JSON, like every other page.",
        "ad_backend" => "AutoEnzyme(Reverse, runtime_activity, Const) — used only " *
                        "to differentiate the reparametrization in the " *
                        "adaptive_centering arm; the bare arms use BridgeStan's " *
                        "own gradient",
        "efficiency_denominator" => "full-run gradient evaluations, including adaptation",
        "ess_reduction" => "minimum over the constrained names the centered and " *
                           "non-centered models SHARE, structurally-constant " *
                           "coordinates dropped and counted in n_constant",
        "representativeness" => "gaussian hierarchical models transcribed by hand " *
                                "from cards in the published BayesianRegressionModels " *
                                "catalogue; every card is parseable=false, so no " *
                                "model here is machine-derived from the catalogue",
        "catalogue" => "https://juliabayes.github.io/BayesianRegressionModels.jl/",
        "controls" => "the noncentered and centered arms carry no reparametrizer, " *
                      "so their nonlinear_adapt=true rows MUST equal their " *
                      "nonlinear_adapt=false rows; that equality is the check " *
                      "that the flag is wired to what the third arm claims",
        "n_seeds" => N_SEEDS,
        "n_draws" => N_DRAWS,
        "seeds" => collect(1:N_SEEDS),
        "unreached" => [Dict("dataset" => u.dataset, "formula" => u.formula,
                             "reason" => u.reason) for u in UNREACHED],
        "generated_at" => string(now()),
    )

    for spec in specs
        t0 = time()
        builder, data = spec.make()
        sb_nc = SBBRMI(builder(data); mod = @__MODULE__)
        sb_c = SBBRMI(builder(data); mod = @__MODULE__, centered_groups = spec.groups)
        p_nc = StanBlocks.stan_instantiate(sb_nc.model)
        p_c = StanBlocks.stan_instantiate(sb_c.model)
        names_nc = BS.param_names(p_nc.model; include_tp = true)
        names_c = BS.param_names(p_c.model; include_tp = true)
        shared = intersect(Set(names_nc), Set(names_c))
        wrapped = BRM.adaptive_centering_problem(sb_nc, p_nc, AD_BACKEND)

        push!(MODEL_META, Dict(
            "spec" => spec.key, "dataset" => spec.dataset,
            "catalogue_source" => spec.source,
            "catalogue_formula" => spec.formula,
            "transcription_note" => spec.note,
            "grouping_factors" => string.(spec.groups),
            "data_url" => SOURCES[spec.dataset],
            "data_sha256" => data_sha(spec.dataset),
            "n_obs" => length(first(data)),
            "dim_noncentered" => LogDensityProblems.dimension(p_nc),
            "dim_centered" => LogDensityProblems.dimension(p_c),
            "n_names_noncentered" => length(names_nc),
            "n_names_centered" => length(names_c),
            "n_names_shared" => length(shared),
        ))

        arms = [("noncentered", p_nc, p_nc.model, names_nc),
                ("centered", p_c, p_c.model, names_c),
                ("adaptive_centering", wrapped, p_nc.model, names_nc)]

        for (arm, prob, model, nms) in arms, adapt in (false, true), seed in 1:N_SEEDS
            push!(ROWS, run_arm(; spec_key = spec.key, arm, problem = prob, model,
                                names = nms, shared, adapt, seed))
        end

        # Write after every spec: a NaN or a crash in spec 7 must not destroy the
        # six specs already measured.
        flush_out(config)
        done = [r for r in ROWS if r.spec == spec.key]
        @info "spec done" spec = spec.key seconds = round(time() - t0, digits = 1) rows = length(done) failed = count(!r.ok for r in done)
        flush(stderr)
    end

    flush_out(config)
    @info "wrote" OUT rows = length(ROWS)
end

main()
