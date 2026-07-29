#!/usr/bin/env julia

# Acceptance gate for the checked-in generated-BRM standard-warmup artifact.
# This validates the complete experimental design before the documentation
# reduces seed rows to tables or plots.

using JSON, SHA

const EXPECTED_SPECS = Set([
    "lme4:dyestuff_re",
    "lme4:sleepstudy_slope",
    "bambi:sleepstudy",
    "mixed_models_jl:penicillin_crossed",
    "bambi:radon_partial",
    "bambi:radon_floor",
    "bambi:radon_slopes",
    "bambi:dietox",
])

const EXPECTED_ARMS = Set([
    "warmuphmc_noncentered",
    "warmuphmc_centered",
    "warmuphmc_fixed_centering",
    "warmuphmc_adaptive_centering",
    "dynamichmc_noncentered",
    "dynamichmc_centered",
])

const EXPECTED_PARAMETERIZATION = Dict(
    "warmuphmc_noncentered" => "noncentered",
    "warmuphmc_centered" => "centered",
    "warmuphmc_fixed_centering" =>
        "adaptive_wrapper_fixed_at_generated_endpoint",
    "warmuphmc_adaptive_centering" => "adaptive_centering",
    "dynamichmc_noncentered" => "noncentered",
    "dynamichmc_centered" => "centered",
)

const EXPECTED_RADON_AUX_SHA256 =
    "5aa648547b9b565d77b9f55defd2f292520441cc6df7a27206802006dade7b63"

function require(condition, message)
    condition || error(message)
end

finite_positive(value) = value isa Real && isfinite(value) && value > 0

function verify(path)
    artifact = JSON.parsefile(path)
    config = artifact["config"]
    models = artifact["models"]
    rows = artifact["rows"]

    require(config["mode"] == "standard", "artifact mode is not standard")
    require(config["host"] == "strato2", "artifact was not measured on strato2")
    require(config["n_seeds"] == 12, "artifact must contain 12 seeds")
    require(config["n_draws"] == 500, "artifact must request 500 retained draws")
    require(config["timing_preflight_draws"] == 50,
            "artifact must use the 50-draw timing preflight")
    require(config["warmuphmc_src_dirty"] == false,
            "WarmupHMC src was dirty during the run")
    require(config["brm_inventory_dirty"] == false,
            "BRM inventory was dirty during the run")
    require(haskey(config, "run_finished_at"), "artifact is only a partial run")
    require(finite_positive(config["total_elapsed_s"]),
            "artifact lacks a positive total elapsed time")

    require(Set(model["spec"] for model in models) == EXPECTED_SPECS,
            "artifact model set does not match the eight audited rows")
    require(length(models) == length(EXPECTED_SPECS),
            "artifact contains duplicate model metadata")
    for model in models
        spec = model["spec"]
        require(model["translation_status"] == "ready",
                "$spec is not a ready inventory translation")
        require(model["surface_support_class"] == "already-expressible-verbatim",
                "$spec is not a verbatim inventory translation")
        require(model["capability_tier"] == "bridgestan-finite-density-gradient",
                "$spec lacks the finite-density/gradient capability tier")
        require(bytes2hex(sha256(model["current_brm_body"])) ==
                    model["current_brm_body_sha256"],
                "$spec generated-body hash is inconsistent")
        if spec in ("bambi:radon_floor", "bambi:radon_slopes")
            require(model["source_fidelity_verdict"] == "adapted-but-defensible",
                    "$spec lost its adapted-but-defensible source label")
        end
        if startswith(spec, "bambi:radon_")
            require(model["auxiliary_data"]["sha256"] == EXPECTED_RADON_AUX_SHA256,
                    "$spec has the wrong pinned cty.dat checksum")
        end
    end

    expected_cells = Set((spec, arm, seed) for spec in EXPECTED_SPECS,
                         arm in EXPECTED_ARMS, seed in 1:12)
    actual_cells = Set((row["spec"], row["arm"], row["seed"]) for row in rows)
    require(length(rows) == length(expected_cells),
            "artifact has $(length(rows)) rows, expected $(length(expected_cells))")
    require(actual_cells == expected_cells,
            "artifact has missing, duplicate, or unexpected model/arm/seed cells")

    for row in rows
        label = "$(row["spec"]) / $(row["arm"]) / seed $(row["seed"])"
        require(row["ok"] == true, "$label failed: $(row["error"])")
        require(isempty(row["error"]), "$label retained a nonempty error")
        require(row["n_draws_actual"] == 500, "$label returned the wrong draw count")
        require(row["n_draws_constrained"] == 500,
                "$label did not constrain all retained draws")
        require(row["grad_evals"] > 0, "$label recorded no gradients")
        require(finite_positive(row["wall_s"]), "$label has invalid wall time")
        require(finite_positive(row["ess_min_shared_constrained"]),
                "$label has invalid shared constrained-space ESS")
        require(finite_positive(row["ess_min_per_grad"]),
                "$label has invalid ESS/gradient")
        require(row["parameterization"] == EXPECTED_PARAMETERIZATION[row["arm"]],
                "$label has the wrong parameterization label")
        expected_sampler = startswith(row["arm"], "dynamichmc_") ?
            "dynamichmc" : "warmuphmc"
        require(row["sampler"] == expected_sampler,
                "$label has the wrong sampler label")
        require(row["nonlinear_adapt"] ==
                    (row["arm"] == "warmuphmc_adaptive_centering"),
                "$label has the wrong nonlinear-adaptation flag")
    end

    digest = bytes2hex(open(sha256, path))
    total_wall = sum(row["wall_s"] for row in rows)
    divergences = sum(row["n_divergent"] for row in rows)
    println("generated-BRM standard artifact OK")
    println("  design: 8 models × 6 arms × 12 seeds = $(length(rows)) rows")
    println("  retained draws: $(sum(row["n_draws_actual"] for row in rows))")
    println("  summed sampling wall time: $(round(total_wall; digits=3)) s")
    println("  process elapsed time: $(config["total_elapsed_s"]) s")
    println("  divergences: $divergences")
    println("  sha256: $digest")
end

path = isempty(ARGS) ? joinpath(
    @__DIR__, "results", "brm_inventory_standard", "rows.json",
) : only(ARGS)
verify(path)
