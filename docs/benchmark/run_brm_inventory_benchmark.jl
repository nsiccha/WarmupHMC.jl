# Benchmark real data through BRM's landed historical-inventory translations.
#
# Unlike `run_brm_catalogue_benchmark.jl`, this runner contains no copied model
# formula. It selects ready rows from BRM's checked-in `translations.tsv`,
# evaluates each row's `current_brm_body` through BRM._brm, and lowers that BRMI
# through SBBRMI. Only the real-data adapters live here because the inventory's
# executable probe currently supplies synthetic data and has no generic
# real-data loader.
#
#   BRMI_SEEDS=12 BRMI_DRAWS=500 julia --startup-file=no \
#     --project=/path/to/pinned/environment \
#     docs/benchmark/run_brm_inventory_benchmark.jl
#
# Set `BRMI_MODE=standard` to compare WarmupHMC with DynamicHMC's default,
# Stan-style warmup on the same generated non-centered/centered targets. That
# mode counts gradients at one shared LogDensityProblems boundary for every
# sampler and runs the full six-arm linear/nonlinear comparison matrix instead
# of comparing sampler-specific internal counters.

using LinearAlgebra
BLAS.set_num_threads(1)

using BayesianRegressionModels, CSV, DataFrames, Dates, Distributions, Downloads
using Enzyme, JSON, LogDensityProblems, MCMCDiagnosticTools, Random, SHA
using StanBlocks, Statistics, WarmupHMC
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const AD_BACKEND = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                                function_annotation=Enzyme.Const)

const N_SEEDS = parse(Int, get(ENV, "BRMI_SEEDS", "12"))
const N_DRAWS = parse(Int, get(ENV, "BRMI_DRAWS", "500"))
const PREFLIGHT_DRAWS = parse(Int, get(ENV, "BRMI_PREFLIGHT_DRAWS", "50"))
const MODE = get(ENV, "BRMI_MODE", "inventory")
MODE in ("inventory", "standard") ||
    error("BRMI_MODE must be inventory or standard, got $(repr(MODE))")
const OUT = get(ENV, "BRMI_OUT", joinpath(
    @__DIR__, "results",
    MODE == "standard" ? "brm_inventory_standard" : "brm_inventory_generated",
    "rows.json"))
const DATA_CACHE = get(ENV, "BRMI_DATA_CACHE", joinpath(tempdir(), "brm-inventory-data"))
const RADON_SRRS2_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/" *
    "5ae7aab3113eb8caa7b95faddb2ecefeaa0f7c9f/examples/data/srrs2.dat"
const RADON_CTY_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/" *
    "5ae7aab3113eb8caa7b95faddb2ecefeaa0f7c9f/examples/data/cty.dat"
const RADON_SRRS2_SHA256 =
    "241219ed05d4bf7dc171ca279c69db7f8d87ae666feae86161c973483a054877"
const RADON_CTY_SHA256 =
    "5aa648547b9b565d77b9f55defd2f292520441cc6df7a27206802006dade7b63"

pkgdir_of(m) = dirname(dirname(pathof(m)))
const INVENTORY_DIR = joinpath(pkgdir_of(BRM), "research", "historical_model_inventory")
const TRANSLATIONS = joinpath(INVENTORY_DIR, "translations.tsv")
const MODEL_MATRIX = joinpath(INVENTORY_DIR, "model_matrix.tsv")

struct InventorySpec
    source::String
    key::String
    dataset::String
    url::String
    adapter_note::String
    adapt::Function
end

function dense_int(v)
    levels = sort(unique(v))
    code = Dict(x => i for (i, x) in enumerate(levels))
    [code[x] for x in v]
end

parse_int(x) = parse(Int, strip(string(x)))

function cached_download(filename, url; expected_sha256=nothing)
    mkpath(DATA_CACHE)
    path = joinpath(DATA_CACHE, filename)
    isfile(path) || Downloads.download(url, path)
    actual = bytes2hex(open(sha256, path))
    isnothing(expected_sha256) || actual == expected_sha256 || error(
        "dataset checksum mismatch for $url: expected $expected_sha256, got $actual",
    )
    path
end

function radon_adapter(srrs2)
    cty_path = cached_download("radon_cty.dat", RADON_CTY_URL;
                               expected_sha256=RADON_CTY_SHA256)
    cty = CSV.read(cty_path, DataFrame)

    mn = copy(srrs2[strip.(String.(srrs2.state)) .== "MN", :])
    mn.log_radon = log.(Float64.(mn.activity) .+ 0.1)
    mn.fips = parse_int.(mn.stfips) .* 1000 .+ parse_int.(mn.cntyfips)
    cty.fips = parse_int.(cty.stfips) .* 1000 .+ parse_int.(cty.ctfips)
    cty.log_u = log.(Float64.(cty.Uppm))
    merged = innerjoin(mn, select(cty, :fips, :log_u); on=:fips)
    unique!(merged, :idnum)

    (; log_radon=Float64.(merged.log_radon),
       floor=dense_int(Int.(merged.floor)),
       county=dense_int(strip.(String.(merged.county))))
end

const SPECS = InventorySpec[
    InventorySpec(
        "lme4", "dyestuff_re", "dyestuff",
        "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv",
        "Yield copied as Float64; Batch deterministically recoded to dense integers",
        df -> (; Yield=Float64.(df.Yield), Batch=dense_int(df.Batch)),
    ),
    InventorySpec(
        "lme4", "sleepstudy_slope", "sleepstudy",
        "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
        "Reaction and Days copied as Float64 without response scaling; Subject " *
        "deterministically recoded to dense integers",
        df -> (; Reaction=Float64.(df.Reaction), Days=Float64.(df.Days),
               Subject=dense_int(df.Subject)),
    ),
    InventorySpec(
        "bambi", "sleepstudy", "sleepstudy",
        "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
        "Reaction and Days copied as Float64 without response scaling; Subject " *
        "deterministically recoded to dense integers",
        df -> (; Reaction=Float64.(df.Reaction), Days=Float64.(df.Days),
               Subject=dense_int(df.Subject)),
    ),
    InventorySpec(
        "mixed_models_jl", "penicillin_crossed", "penicillin",
        "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv",
        "diameter copied as Float64; plate and sample converted to String and " *
        "deterministically recoded to dense integers",
        df -> (; diameter=Float64.(df.diameter),
               plate=dense_int(String.(df.plate)),
               sample=dense_int(String.(df.sample))),
    ),
    InventorySpec(
        "bambi", "radon_partial", "radon",
        RADON_SRRS2_URL,
        "Pinned historical two-file loader: Minnesota rows from srrs2.dat; " *
        "log_radon=log(activity+0.1); FIPS join to cty.dat; unique idnum; " *
        "floor and stripped county densely recoded. Auxiliary cty.dat sha256=" *
        RADON_CTY_SHA256,
        radon_adapter,
    ),
    InventorySpec(
        "bambi", "radon_floor", "radon",
        RADON_SRRS2_URL,
        "Pinned historical two-file loader: Minnesota rows from srrs2.dat; " *
        "log_radon=log(activity+0.1); FIPS join to cty.dat; unique idnum; " *
        "floor and stripped county densely recoded. Auxiliary cty.dat sha256=" *
        RADON_CTY_SHA256,
        radon_adapter,
    ),
    InventorySpec(
        "bambi", "radon_slopes", "radon",
        RADON_SRRS2_URL,
        "Pinned historical two-file loader: Minnesota rows from srrs2.dat; " *
        "log_radon=log(activity+0.1); FIPS join to cty.dat; unique idnum; " *
        "floor and stripped county densely recoded. Auxiliary cty.dat sha256=" *
        RADON_CTY_SHA256,
        radon_adapter,
    ),
    InventorySpec(
        "bambi", "dietox", "dietox",
        "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv",
        "Weight and Time copied as Float64; Pig deterministically recoded to dense integers",
        df -> (; Weight=Float64.(df.Weight), Time=Float64.(df.Time),
               Pig=dense_int(df.Pig)),
    ),
]

selected_specs() = haskey(ENV, "BRMI_SPECS") ?
    [spec for spec in SPECS
     if "$(spec.source):$(spec.key)" in split(ENV["BRMI_SPECS"], ',')] : SPECS

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")

function read_tsv(path)
    lines = readlines(path)
    columns = split(first(lines), '\t'; keepempty=true)
    [Dict(columns .=> tsv_unescape.(split(line, '\t'; keepempty=true)))
     for line in Iterators.drop(lines, 1)]
end

const TRANSLATION_ROWS = read_tsv(TRANSLATIONS)
const MATRIX_ROWS = read_tsv(MODEL_MATRIX)

function inventory_row(spec)
    row = only(r for r in TRANSLATION_ROWS
               if r["source"] == spec.source && r["key"] == spec.key &&
                  r["variant"] == "inferred-family")
    row["translation_status"] == "ready" ||
        error("$(spec.source):$(spec.key) is not a ready translation")
    row["surface_support_class"] == "already-expressible-verbatim" ||
        error("$(spec.source):$(spec.key) is not a verbatim translation")

    matrix = only(r for r in MATRIX_ROWS
                  if r["row_key"] == "$(spec.source):$(spec.key)")
    matrix["inferred_translation_status"] == "ready" ||
        error("authoritative matrix does not mark $(spec.source):$(spec.key) ready")
    matrix["inferred_surface_support_class"] == "already-expressible-verbatim" ||
        error("authoritative matrix does not mark $(spec.source):$(spec.key) verbatim")
    matrix["inferred_capability_tier"] == "bridgestan-finite-density-gradient" ||
        error("authoritative matrix lacks a finite-gradient receipt for $(spec.source):$(spec.key)")
    matrix["inferred_current_brm_body"] == row["current_brm_body"] ||
        error("matrix/translation body disagreement for $(spec.source):$(spec.key)")
    row, matrix
end

function dataset(spec)
    expected = spec.dataset == "radon" ? RADON_SRRS2_SHA256 : nothing
    cached_download(spec.dataset * ".csv", spec.url; expected_sha256=expected)
end

function materialize(row, data; centered_groups=Symbol[])
    body = row["current_brm_body"]
    brmi = Core.eval(@__MODULE__, BRM._brm(body; df=data))
    sb = SBBRMI(brmi; mod=@__MODULE__, centered_groups)
    name = Symbol(replace(row["source"] * "_" * row["key"], r"\W" => "_"))
    descriptor = brm_descriptor(sb; name)
    (; brmi, sb, descriptor)
end

gitsha(dir) = try readchomp(`git -C $dir rev-parse HEAD`) catch; "unavailable" end
gitdirty(dir, paths...) = try
    !isempty(readchomp(`git -C $dir status --porcelain -- $(collect(paths))`))
catch
    true
end

function ess_vec(draws::AbstractMatrix)
    d, n = size(draws)
    (n < 10 || !all(isfinite, draws)) && return Float64[]
    collect(MCMCDiagnosticTools.ess(reshape(permutedims(draws), (n, 1, d))))
end

function finite_min(v)
    w = filter(isfinite, v)
    isempty(w) ? (NaN, length(v)) : (minimum(w), length(v) - length(w))
end

function shared_constrained_ess(model, names, shared, draws)
    keep = [i for (i, name) in enumerate(names) if name in shared]
    out = Matrix{Float64}(undef, length(keep), size(draws, 2))
    ok = falses(size(draws, 2))
    for j in axes(draws, 2)
        try
            values = BS.param_constrain(model, Vector{Float64}(draws[:, j]); include_tp=true)
            out[:, j] = values[keep]
            ok[j] = all(isfinite, @view out[:, j])
        catch
            ok[j] = false
        end
    end
    minimum_ess, n_constant = finite_min(ess_vec(out[:, ok]))
    minimum_ess, count(ok), n_constant
end

function run_arm(; spec_key, arm, problem, model, names, shared, adapt, seed)
    try
        wall = @elapsed result = adaptive_warmup_mcmc(
            Xoshiro(seed), problem; n_draws=N_DRAWS,
            nonlinear_adapt=adapt, progress=nothing,
        )
        draws = Matrix{Float64}(result.posterior_position)
        ess_unc, _ = finite_min(ess_vec(draws))
        ess_con, n_ok, n_constant = shared_constrained_ess(
            model, names, shared, draws)
        grad = Int(result.total_evaluation_counter)
        (; spec=spec_key, arm, nonlinear_adapt=adapt, seed, ok=true,
         wall_s=wall, grad_evals=grad, ess_min_unconstrained=ess_unc,
         ess_min_shared_constrained=ess_con,
         ess_min_per_grad=ess_con / max(grad, 1),
         n_draws_constrained=n_ok, n_constant,
         n_divergent=Int(result.n_divergent_samples), error="")
    catch err
        (; spec=spec_key, arm, nonlinear_adapt=adapt, seed, ok=false,
         wall_s=NaN, grad_evals=0, ess_min_unconstrained=NaN,
         ess_min_shared_constrained=NaN, ess_min_per_grad=NaN,
         n_draws_constrained=0, n_constant=0, n_divergent=0,
         error=first(sprint(showerror, err, catch_backtrace()), 1200))
    end
end

function sample_standard(sampler, rng, problem, n_draws; nonlinear_adapt=true)
    if sampler == "warmuphmc"
        result = adaptive_warmup_mcmc(
            rng, problem; n_draws, nonlinear_adapt, progress=nothing,
        )
        Matrix{Float64}(result.posterior_position),
            Int(result.n_divergent_samples)
    elseif sampler == "dynamichmc"
        result = WarmupHMC.DynamicHMC.mcmc_with_warmup(
            rng, problem, n_draws;
            reporter=WarmupHMC.DynamicHMC.NoProgressReport(),
        )
        ndiv = count(
            s -> WarmupHMC.DynamicHMC.is_divergent(s.termination),
            result.tree_statistics,
        )
        Matrix{Float64}(result.posterior_matrix), ndiv
    else
        error("unknown standard-comparison sampler $(repr(sampler))")
    end
end

"""
Run one standard-warmup comparison arm.

`base_problem` is always the generated BRM density. `build_problem` receives
the externally counted version of that density, so the adaptive-centering arm
keeps its `ReparametrizedProblem` outer type while every sampler is counted at
the same inner logdensity-and-gradient boundary.
"""
function run_standard_arm(; spec_key, arm, sampler, parameterization,
                            base_problem, build_problem, model, names, shared,
                            nonlinear_adapt, seed, n_draws=N_DRAWS)
    try
        timed = WarmupHMC.count_and_time(base_problem) do counted
            problem = build_problem(counted)
            sample_standard(sampler, Xoshiro(seed), problem, n_draws;
                            nonlinear_adapt)
        end
        draws, ndiv = timed.result
        ess_unc, _ = finite_min(ess_vec(draws))
        ess_con, n_ok, n_constant = shared_constrained_ess(
            model, names, shared, draws)
        grad = Int(timed.n_evaluations)
        (; spec=spec_key, arm, sampler, parameterization, nonlinear_adapt,
         seed, ok=true,
         wall_s=timed.elapsed, grad_evals=grad,
         n_draws_actual=size(draws, 2),
         ess_min_unconstrained=ess_unc,
         ess_min_shared_constrained=ess_con,
         ess_min_per_grad=ess_con / max(grad, 1),
         n_draws_constrained=n_ok, n_constant,
         n_divergent=Int(ndiv), error="")
    catch err
        (; spec=spec_key, arm, sampler, parameterization, nonlinear_adapt,
         seed, ok=false,
         wall_s=NaN, grad_evals=0, n_draws_actual=0,
         ess_min_unconstrained=NaN,
         ess_min_shared_constrained=NaN, ess_min_per_grad=NaN,
         n_draws_constrained=0, n_constant=0, n_divergent=0,
         error=first(sprint(showerror, err, catch_backtrace()), 1200))
    end
end

sanitize(x) = x isa AbstractFloat && !isfinite(x) ? nothing : x
sanitize(x::AbstractDict) = Dict(k => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]

const ROWS = Any[]
const MODEL_META = Any[]

function flush_out(config)
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        JSON.print(io, Dict(
            "config" => sanitize(config),
            "models" => sanitize(MODEL_META),
            "rows" => [sanitize(Dict(string(k) => v for (k, v) in pairs(row)))
                       for row in ROWS],
        ), 2)
    end
end

function main()
    brm_dir = pkgdir_of(BRM)
    whmc_dir = pkgdir_of(WarmupHMC)
    reproduction = MODE == "standard" ?
        "BRMI_MODE=standard BRMI_SEEDS=$(N_SEEDS) BRMI_DRAWS=$(N_DRAWS) " *
        "BRMI_PREFLIGHT_DRAWS=$(PREFLIGHT_DRAWS) julia --startup-file=no " *
        "--project=/path/to/pinned/environment " *
        "docs/benchmark/run_brm_inventory_benchmark.jl" :
        "BRMI_SEEDS=$(N_SEEDS) BRMI_DRAWS=$(N_DRAWS) " *
        "BRMI_PREFLIGHT_DRAWS=$(PREFLIGHT_DRAWS) julia --startup-file=no " *
        "--project=/path/to/pinned/environment " *
        "docs/benchmark/run_brm_inventory_benchmark.jl"
    run_order = MODE == "standard" ?
        "one untimed preflight per comparison arm; the six recorded arms " *
        "rotate cyclically by seed to avoid systematic temporal bias" :
        "one untimed preflight per arm/flag; recorded flag order alternates " *
        "by seed to avoid systematic temporal bias"
    config = Dict(
        "host" => get(ENV, "KB_HOST", gethostname()),
        "julia" => string(VERSION),
        "blas_threads" => BLAS.get_num_threads(),
        "mode" => MODE,
        "sampler" => MODE == "standard" ?
            "adaptive_warmup_mcmc vs DynamicHMC.mcmc_with_warmup" :
            "adaptive_warmup_mcmc",
        "runner" => "docs/benchmark/run_brm_inventory_benchmark.jl",
        "reproduction" => reproduction,
        "warmuphmc_sha" => gitsha(whmc_dir),
        "dynamichmc_version" => string(Base.pkgversion(WarmupHMC.DynamicHMC)),
        "brm_sha" => gitsha(brm_dir),
        "stanblocks_sha" => gitsha(pkgdir_of(StanBlocks)),
        "warmuphmc_src_dirty" => gitdirty(whmc_dir, "src"),
        "brm_inventory_dirty" => gitdirty(brm_dir, "research/historical_model_inventory"),
        "translations_sha256" => bytes2hex(open(sha256, TRANSLATIONS)),
        "model_matrix_sha256" => bytes2hex(open(sha256, MODEL_MATRIX)),
        "inventory_contract" => "model bodies are read from translations.tsv and " *
            "cross-checked against model_matrix.tsv; real-data adapters are consumer-side",
        "ad_backend" => "AutoEnzyme(Reverse, runtime_activity, Const), used by " *
                        "BRM.adaptive_centering_problem",
        "controls" => "the six-arm standard matrix separates WarmupHMC's " *
            "linear adaptation on generated non-centered/centered targets from " *
            "the fixed and fitted nonlinear-wrapper paths",
        "n_seeds" => N_SEEDS,
        "n_draws" => N_DRAWS,
        "timing_preflight_draws" => PREFLIGHT_DRAWS,
        "run_order" => run_order,
        "gradient_counter" => MODE == "standard" ?
            "WarmupHMC.count_and_time around the generated BRM density for " *
            "every sampler; the adaptive wrapper is built over the counted inner target" :
            "adaptive_warmup_mcmc total_evaluation_counter",
        "warmup_budget" => MODE == "standard" ?
            "sampler defaults: DynamicHMC uses its default 1000-step Stan-style " *
            "warmup; WarmupHMC chooses its own gradient-targeted windows" : nothing,
        "seeds" => collect(1:N_SEEDS),
        "generated_at" => string(now()),
    )

    for spec in selected_specs()
        started = time()
        row, matrix = inventory_row(spec)
        path = dataset(spec)
        df = CSV.read(path, DataFrame)
        data = spec.adapt(df)
        groups = Symbol.(filter(value -> !isempty(value), split(row["group_columns"], ',')))

        built_nc = materialize(row, data)
        built_c = materialize(row, data; centered_groups=groups)
        problem_nc = StanBlocks.stan_instantiate(built_nc.sb.model)
        problem_c = StanBlocks.stan_instantiate(built_c.sb.model)
        names_nc = BS.param_names(problem_nc.model; include_tp=true)
        names_c = BS.param_names(problem_c.model; include_tp=true)
        unc_names_nc = BS.param_unc_names(problem_nc.model)
        shared = intersect(Set(names_nc), Set(names_c))
        descriptor_code = brm_execute(built_nc.descriptor, :transpile)

        spec_key = "$(spec.source):$(spec.key)"
        push!(MODEL_META, Dict(
            "spec" => spec_key,
            "source" => spec.source,
            "key" => spec.key,
            "variant" => row["variant"],
            "translation_status" => row["translation_status"],
            "surface_support_class" => row["surface_support_class"],
            "semantic_route" => row["semantic_route"],
            "capability_tier" => matrix["inferred_capability_tier"],
            "probe_evidence_kind" => matrix["probe_evidence_kind"],
            "probe_id" => matrix["probe_id"],
            "source_fidelity_verdict" => get(matrix, "source_fidelity_verdict", ""),
            "source_fidelity_reason" => get(matrix, "source_fidelity_reason", ""),
            "source_fidelity_manual_reviewed" =>
                get(matrix, "source_fidelity_manual_reviewed", ""),
            "inferred_family" => get(matrix, "inferred_family", ""),
            "inferred_family_provenance" =>
                get(matrix, "inferred_family_provenance", ""),
            "family_support" => get(matrix, "family_support", ""),
            "dataset_support" => get(matrix, "dataset_support", ""),
            "dataset_receipt_urls" => get(matrix, "dataset_receipt_urls", ""),
            "row_source_claim" => get(matrix, "row_source_claim", ""),
            "historical_formula" => row["formula_claim"],
            "current_brm_body" => row["current_brm_body"],
            "current_brm_body_sha256" => bytes2hex(sha256(row["current_brm_body"])),
            "grouping_factors" => string.(groups),
            "dataset" => spec.dataset,
            "data_url" => spec.url,
            "data_sha256" => bytes2hex(open(sha256, path)),
            "auxiliary_data" => spec.dataset == "radon" ? Dict(
                "url" => RADON_CTY_URL,
                "sha256" => RADON_CTY_SHA256,
            ) : Dict{String,String}(),
            "data_adapter" => spec.adapter_note,
            "n_obs" => length(first(data)),
            "descriptor_operations" => string.(getproperty.(built_nc.descriptor.operations, :name)),
            "descriptor_stan_sha256" => bytes2hex(sha256(descriptor_code)),
            "dim_noncentered" => LogDensityProblems.dimension(problem_nc),
            "dim_centered" => LogDensityProblems.dimension(problem_c),
            "n_names_noncentered" => length(names_nc),
            "n_names_centered" => length(names_c),
            "n_names_shared" => length(shared),
        ))

        problem_for(arm) = if arm == "noncentered"
            (problem_nc, problem_nc.model, names_nc)
        elseif arm == "centered"
            (problem_c, problem_c.model, names_c)
        else
            # Rebuild the wrapper for every run so one seed's adaptive
            # centering cannot become the next seed's starting point.
            (BRM.adaptive_centering_problem(
                built_nc.sb, problem_nc, AD_BACKEND), problem_nc.model, names_nc)
        end

        if MODE == "inventory"
            # Keep Julia/Enzyme compilation out of the sampling-time comparison.
            # Both Boolean paths are exercised because the runtime flag controls a
            # different warm-up branch even though its type is the same.
            for arm in ("noncentered", "centered", "adaptive_centering"),
                adapt in (false, true)
                problem, _, _ = problem_for(arm)
                adaptive_warmup_mcmc(
                    Xoshiro(0x6b625000 + 10 * findfirst(==(arm),
                        ("noncentered", "centered", "adaptive_centering")) + adapt),
                    problem; n_draws=PREFLIGHT_DRAWS,
                    nonlinear_adapt=adapt, progress=nothing,
                )
            end

            for arm in ("noncentered", "centered", "adaptive_centering"),
                seed in 1:N_SEEDS
                adapt_order = isodd(seed) ? (false, true) : (true, false)
                for adapt in adapt_order
                    problem, model, names = problem_for(arm)
                    push!(ROWS, run_arm(; spec_key, arm, problem, model, names, shared,
                                        adapt, seed))
                end
            end
        else
            identity_problem = counted -> counted
            adaptive_problem = counted -> BRM.adaptive_centering_problem(
                built_nc.sb, counted, AD_BACKEND; unc_names=unc_names_nc)
            standard_arms = [
                (; arm="warmuphmc_noncentered", sampler="warmuphmc",
                 parameterization="noncentered", base_problem=problem_nc,
                 build_problem=identity_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=false),
                (; arm="warmuphmc_centered", sampler="warmuphmc",
                 parameterization="centered", base_problem=problem_c,
                 build_problem=identity_problem, model=problem_c.model,
                 names=names_c, nonlinear_adapt=false),
                (; arm="warmuphmc_fixed_centering", sampler="warmuphmc",
                 parameterization="adaptive_wrapper_fixed_at_generated_endpoint",
                 base_problem=problem_nc, build_problem=adaptive_problem,
                 model=problem_nc.model, names=names_nc,
                 nonlinear_adapt=false),
                (; arm="warmuphmc_adaptive_centering", sampler="warmuphmc",
                 parameterization="adaptive_centering", base_problem=problem_nc,
                 build_problem=adaptive_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=true),
                (; arm="dynamichmc_noncentered", sampler="dynamichmc",
                 parameterization="noncentered", base_problem=problem_nc,
                 build_problem=identity_problem, model=problem_nc.model,
                 names=names_nc, nonlinear_adapt=false),
                (; arm="dynamichmc_centered", sampler="dynamichmc",
                 parameterization="centered", base_problem=problem_c,
                 build_problem=identity_problem, model=problem_c.model,
                 names=names_c, nonlinear_adapt=false),
            ]

            for (i, a) in enumerate(standard_arms)
                preflight = run_standard_arm(;
                    spec_key, a..., shared,
                    seed=0x6b645000 + i, n_draws=PREFLIGHT_DRAWS,
                )
                preflight.ok || error(
                    "standard comparison preflight failed for $(a.arm): $(preflight.error)")
            end

            for seed in 1:N_SEEDS
                offset = mod(seed - 1, length(standard_arms))
                order = vcat(standard_arms[offset+1:end], standard_arms[1:offset])
                for a in order
                    push!(ROWS, run_standard_arm(; spec_key, a..., shared, seed))
                end
            end
        end

        flush_out(config)
        completed = [r for r in ROWS if r.spec == spec_key]
        elapsed_seconds = round(time() - started; digits=1)
        @info "inventory spec done" spec=spec_key seconds=elapsed_seconds rows=length(completed) failed=count(!r.ok for r in completed)
        flush(stderr)
    end

    flush_out(config)
    @info "wrote inventory-generated benchmark" OUT rows=length(ROWS)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
