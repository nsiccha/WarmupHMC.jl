#!/usr/bin/env julia

# Acceptance gate for the checked-in generated-BRM standard-warmup artifact.
# This validates the complete experimental design before the documentation
# reduces seed rows to tables or plots.

using JSON, SHA

# The eight controls the published matrix started with, then the
# historical-gallery tranche. Spelling the set out is the point: a spec that
# silently stops being benchmarked — because its download broke, or an inventory
# edit dropped it below the readiness gate — must fail this gate rather than
# shrink the published matrix without saying so.
const EXPECTED_CONTROL_SPECS = Set([
    "lme4:dyestuff_re",
    "lme4:sleepstudy_slope",
    "bambi:sleepstudy",
    "mixed_models_jl:penicillin_crossed",
    "bambi:radon_partial",
    "bambi:radon_floor",
    "bambi:radon_slopes",
    "bambi:dietox",
])

const EXPECTED_TRANCHE_SPECS = Set([
    "vasishth:meta_sbi",
    "kruschke:fruitfly_anhecova",
    "burkner_papers:epilepsy_simple",
    "kruschke:therapeutic_touch",
    "bambi:hierarchical_binomial_partial",
    "mixed_models_jl:contraception_glmm",
    "bambi:predict_new_groups",
    "vasishth:n400_crossed",
])

const EXPECTED_SPECS = union(EXPECTED_CONTROL_SPECS, EXPECTED_TRANCHE_SPECS)

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

# Read back from BRM's own matrix, not invented here. The closed set is asserted
# for every row so a blank or newly-spelled verdict fails rather than rendering
# as an empty cell; the three `adapted-but-defensible` rows are named because the
# documentation page reports that caveat per row, so an upstream re-verdict has
# to force a look at the prose rather than silently contradict it.
const EXPECTED_FIDELITY_VERDICTS =
    Set(["confirmed", "adapted-but-defensible", "unverifiable"])

const EXPECTED_ADAPTED_SPECS = Set([
    "bambi:radon_floor",
    "bambi:radon_slopes",
    "mixed_models_jl:contraception_glmm",
])

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

    require(haskey(config, "runner_sha256"), "artifact does not pin the runner")
    require(length(config["runner_sha256"]) == 64 &&
            all(c -> c in "0123456789abcdef", config["runner_sha256"]),
            "artifact runner_sha256 is not a sha256 digest")

    require(Set(model["spec"] for model in models) == EXPECTED_SPECS,
            "artifact model set does not match the audited rows")
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

        require(model["source_fidelity_verdict"] in EXPECTED_FIDELITY_VERDICTS,
                "$spec carries an unrecognised source-fidelity verdict " *
                repr(model["source_fidelity_verdict"]))
        require((model["source_fidelity_verdict"] == "adapted-but-defensible") ==
                    (spec in EXPECTED_ADAPTED_SPECS),
                "$spec changed its adapted-but-defensible source label; the " *
                "documentation reports this per row")
        if startswith(spec, "bambi:radon_")
            require(model["auxiliary_data"]["sha256"] == EXPECTED_RADON_AUX_SHA256,
                    "$spec has the wrong pinned cty.dat checksum")
        end

        # The data pin. `data_sha256` is what the file on disk actually hashed to
        # and `data_sha256_pinned` is what the spec demanded; the runner already
        # refuses a mismatch, so this is the checked-in evidence that it did.
        require(!isempty(model["data_sha256_pinned"]),
                "$spec has no pinned upstream data checksum")
        require(model["data_sha256_pinned"] == model["data_sha256"],
                "$spec data checksum does not match its pin")

        require(!isempty(model["coverage"]),
                "$spec has no coverage rationale; every published row must say " *
                "what it is there to cover")
        require(!isempty(model["data_adapter"]),
                "$spec has no adapter note")

        # Block widths under both readings. These are what the page's
        # narrower-than-historical annotation is computed from, so an artifact
        # that lost them would render that section empty and say nothing.
        for field in ("random_effect_blocks", "historical_random_effect_blocks")
            blocks = model[field]
            require(blocks isa AbstractVector && !isempty(blocks),
                    "$spec recorded no $field")
            for block in blocks
                for key in ("k", "terms", "group", "correlated")
                    require(haskey(block, key),
                            "$spec $field entry is missing `$key`")
                end
                require(block["k"] isa Integer && block["k"] >= 0,
                        "$spec $field entry has a nonsensical width")
            end
        end
        require(length(model["random_effect_blocks"]) ==
                    length(model["historical_random_effect_blocks"]),
                "$spec generated and historical block counts disagree, so the " *
                "per-block comparison cannot be positional")

        groups = model["grouping_factors"]
        require(!isempty(groups), "$spec has no grouping factor")
        require(issubset(Set(groups), Set(model["inventory_group_columns"])),
                "$spec fits a grouping factor the inventory body does not name")
        require(Set(keys(model["n_groups"])) == Set(groups),
                "$spec group-level counts do not cover its grouping factors")
        for (group, count) in model["n_groups"]
            require(count isa Integer && count >= 2,
                    "$spec grouping factor `$group` has $count level(s); a " *
                    "hierarchy over fewer than two is not one")
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
        require(row["parameterization"] == EXPECTED_PARAMETERIZATION[row["arm"]],
                "$label has the wrong parameterization label")
        expected_sampler = startswith(row["arm"], "dynamichmc_") ?
            "dynamichmc" : "warmuphmc"
        require(row["sampler"] == expected_sampler,
                "$label has the wrong sampler label")
        require(row["nonlinear_adapt"] ==
                    (row["arm"] == "warmuphmc_adaptive_centering"),
                "$label has the wrong nonlinear-adaptation flag")
        if row["ok"]
            require(isempty(row["error"]), "$label retained a nonempty error")
            require(row["n_draws_actual"] == 500,
                    "$label returned the wrong draw count")
            require(row["n_draws_constrained"] == 500,
                    "$label did not constrain all retained draws")
            require(row["grad_evals"] > 0, "$label recorded no gradients")
            require(finite_positive(row["wall_s"]), "$label has invalid wall time")
            require(finite_positive(row["ess_min_shared_constrained"]),
                    "$label has invalid shared constrained-space ESS")
            require(finite_positive(row["ess_min_per_grad"]),
                    "$label has invalid ESS/gradient")
        else
            require(!isempty(row["error"]),
                    "$label is failed without an explicit diagnostic")
            if row["n_draws_actual"] > 0
                require(row["n_draws_actual"] == 500,
                        "$label retained a partial, unlabelled draw set")
                require(row["grad_evals"] > 0,
                        "$label retained draws but no gradient count")
                require(finite_positive(row["wall_s"]),
                        "$label retained draws but no wall time")
                require(!finite_positive(row["ess_min_shared_constrained"]) ||
                        !finite_positive(row["ess_min_per_grad"]),
                        "$label is marked failed despite finite efficiency diagnostics")
            end
        end
    end

    digest = bytes2hex(open(sha256, path))
    total_wall = sum(row["wall_s"] for row in rows)
    divergences = sum(row["n_divergent"] for row in rows)
    failures = [row for row in rows if !row["ok"]]
    println("generated-BRM standard artifact OK")
    println("  design: $(length(models)) models × $(length(EXPECTED_ARMS)) arms " *
            "× $(config["n_seeds"]) seeds = $(length(rows)) rows")
    println("  controls: $(length(EXPECTED_CONTROL_SPECS)), " *
            "historical-gallery tranche: $(length(EXPECTED_TRANCHE_SPECS))")
    println("  successful rows: $(length(rows) - length(failures))/$(length(rows))")
    for row in failures
        println("  explicit failure: $(row["spec"]) / $(row["arm"]) / " *
                "seed $(row["seed"]): $(row["error"])")
    end
    println("  retained draws: $(sum(row["n_draws_actual"] for row in rows))")
    println("  summed sampling wall time: $(round(total_wall; digits=3)) s")
    println("  process elapsed time: $(config["total_elapsed_s"]) s")
    println("  divergences: $divergences")

    # Per-spec degeneracy, so a report can name which rows were degenerate rather
    # than quote one pooled total. `n_constant` is the count of constrained
    # coordinates that were dropped from the ESS minimum for being constant or
    # non-finite — a nonzero value is not a failure (a `sigma` pinned at a
    # boundary still samples), but it narrows what the published ESS is a
    # minimum over, so it is reported per row rather than summed away.
    println("  per-model degeneracy (divergences / max dropped coords / failed cells):")
    for spec in sort(collect(EXPECTED_SPECS))
        spec_rows = [row for row in rows if row["spec"] == spec]
        println("    $(rpad(spec, 38)) " *
                "$(sum(row["n_divergent"] for row in spec_rows)) / " *
                "$(maximum(row["n_constant"] for row in spec_rows)) / " *
                "$(count(row -> !row["ok"], spec_rows))")
    end
    println("  sha256: $digest")
end

path = isempty(ARGS) ? joinpath(
    @__DIR__, "results", "brm_inventory_standard", "rows.json",
) : only(ARGS)
verify(path)
