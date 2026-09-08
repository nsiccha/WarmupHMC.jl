# The committed public surface, pinned so that changing it is a deliberate act.
#
# WHY A LITERAL LIST HERE, WHEN `test/readme.jl` DELIBERATELY REFUSES ONE
#
# These two files look like they contradict each other. They do not — they point
# in opposite directions on purpose:
#
#   * `test/readme.jl` asks the MODULE what it exports (`names(WarmupHMC)`) and
#     checks the README against it. A literal list there would defeat the point:
#     the README must follow the code, so the code has to be the source of truth.
#
#   * this file IS the contract. The lists below are the promise. Their whole
#     value is that adding an export, removing one, or renaming a returned field
#     cannot happen as a side effect of some other change — it must come here and
#     be edited on purpose, which is exactly the moment to ask whether it is a
#     breaking change.
#
# So: a failure here is NOT "the test is out of date, go fix the list". It is the
# question "is this a breaking change?" being asked at the only moment anyone can
# still answer it cheaply. After 1.0, an unintended change to either list is a
# major version bump that shipped by accident.
#
# WHAT IS AND IS NOT COVERED
#
# Covered: the exported names below, the FIELD SET of the NamedTuple returned by
# `adaptive_warmup_mcmc`, and the keywords it accepts. Not covered: everything
# under `WarmupHMC.` that is not exported (`docs/src/api.md` "Internals" says the
# same), the element types and array types of those fields, and the numeric
# values themselves — those are sampler behaviour, not API.

# ---------------------------------------------------------------------------
# 1. The exported surface
# ---------------------------------------------------------------------------

const PUBLIC_EXPORTS = [
    # samplers
    :adaptive_warmup_mcmc,
    :clustered_warmup_mcmc,
    :cooperative_warmup_mcmc,
    :completion_warmup_mcmc,
    :resume_warmup_mcmc,        # deprecated in favour of `resume=true`, still exported
    # fixed-kernel interruptible streaming sampler
    :stream_mcmc,
    :open_stream,
    # reparametrization types
    :CandidateScoringPlan,
    :IndexedReparametrization,
    :PartiallyCentered,
    :Reparametrization,
    :ReparametrizedProblem,
]

@testset "exported surface is exactly the committed one" begin
    actual = sort(filter(!=(:WarmupHMC), names(WarmupHMC)))
    expected = sort(PUBLIC_EXPORTS)

    # Named diffs, so a failure says WHICH symbol moved rather than dumping two
    # sorted vectors at the reader.
    added = setdiff(actual, expected)
    removed = setdiff(expected, actual)
    isempty(added) || @error "exported but NOT in the committed public API" added
    isempty(removed) || @error "in the committed public API but NO LONGER exported" removed

    @test isempty(added)
    @test isempty(removed)
    @test actual == expected
end

# ---------------------------------------------------------------------------
# 2. The return schema of `adaptive_warmup_mcmc`
# ---------------------------------------------------------------------------
#
# Built at exactly one place — `finalize_warmup!(state::AWMState)` in
# `src/adaptive_warmup_mcmc.jl`. Pinned here against a real run rather than
# against that source text, because what a caller can rely on is what the
# function RETURNS, not how it is spelled.

const PUBLIC_RESULT_FIELDS = [
    :initial_position,
    :halo_position,
    :halo_gradient,
    :posterior_position,
    :posterior_gradient,
    :ess,
    :scale_options,
    :active_transformation,
    :linear_restart_source,
    :linear_trajectory_weighting,
    :linear_metric_fallback,
    :linear_metric_fallbacks,
    :stepsize,
    :total_evaluation_counter,
    :n_divergent_samples,
    :position_and_gradient,
    :scale_changes,
]

@testset "adaptive_warmup_mcmc return schema is exactly the committed one" begin
    problem = DiagGaussian([0.0, 1.0], [1.0, 2.0])
    result = adaptive_warmup_mcmc(Xoshiro(20260728), problem; n_draws=100, progress=nothing)

    actual = collect(propertynames(result))
    added = setdiff(actual, PUBLIC_RESULT_FIELDS)
    removed = setdiff(PUBLIC_RESULT_FIELDS, actual)
    isempty(added) || @error "returned but NOT in the committed schema" added
    isempty(removed) || @error "in the committed schema but NO LONGER returned" removed

    @test isempty(added)
    @test isempty(removed)

    # Order is part of the pin too: the NamedTuple's type is its names AND their
    # order, so a reordering is a type change even when the field set matches.
    @test actual == PUBLIC_RESULT_FIELDS

    # The one field with a documented shape: `posterior_position` is
    # dimension × n_draws, and the README and index.md both say so.
    @test size(result.posterior_position) == (LogDensityProblems.dimension(problem), 100)
end

# ---------------------------------------------------------------------------
# 3. The keywords `adaptive_warmup_mcmc` accepts
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT THE SAME CHECK AS THE ONE IN `web/src/test/kwarg_validation.jl`
#
# That one is a DRIFT guard: it asserts `_SAMPLER_KWARGS` still covers what the
# live method signatures declare, so adding a kwarg without updating the table
# fails there. But it compares two things that MOVE TOGETHER, and it has no
# opinion about which names should exist. Measured, not assumed: renaming
# `variance_cond_target` to `variance_condition_target` throughout `src/` — a
# breaking change to a documented keyword — leaves that guard at 38/38 green,
# and the whole web suite item with it. Correct for what it guards; exactly what
# a semver contract cannot be built on.
#
# This is the missing half: the names themselves, written down once, as a
# promise. Renaming `n_draws` is a breaking change no matter how consistently it
# is renamed.

const PUBLIC_ADAPTIVE_KWARGS = [
    # how much to sample
    :n_draws, :n_evaluations, :recording_target,
    # NUTS / step-size adaptation
    :stepsize_adaptation_limit, :target_acceptance_rate, :max_tree_depth,
    :variance_cond_target,
    # linear adaptation
    :linear_restart_source, :linear_trajectory_weighting, :linear_metric_fallback,
    # nonlinear adaptation
    :nonlinear_adapt, :nonlinear_evidence, :nonlinear_trajectory_weighting,
    :nonlinear_good_leaf_threshold,
    # starting point, reporting, observation
    :init, :progress, :description, :monitor_ess, :callback,
    # checkpointing
    :checkpoint_dir, :resume, :overwrite,
    # deliberate passthrough to the Pathfinder initializer
    :pathfinder_kw,
]

# Named in this package's own source and therefore forwarded on purpose; every
# entry point accepts them on top of its own set. Part of the promise too — a
# caller passing `ntries` is passing a documented keyword, not getting lucky.
const PUBLIC_INITIALIZER_KWARGS = [
    :ntries, :maxiters, :ndraws, :ndraws_elbo, :history_length, :optimizer,
]

@testset "adaptive_warmup_mcmc accepted keywords are exactly the committed ones" begin
    actual_own = collect(WarmupHMC._SAMPLER_KWARGS[:adaptive_warmup_mcmc])
    added = setdiff(actual_own, PUBLIC_ADAPTIVE_KWARGS)
    removed = setdiff(PUBLIC_ADAPTIVE_KWARGS, actual_own)
    isempty(added) || @error "accepted but NOT in the committed keyword set" added
    isempty(removed) || @error "in the committed keyword set but NO LONGER accepted" removed
    @test isempty(added)
    @test isempty(removed)

    @test sort(collect(WarmupHMC._INITIALIZER_KWARGS)) == sort(PUBLIC_INITIALIZER_KWARGS)

    # Guard the guard, and the part that actually matters to a caller: the lists
    # above are a promise only if something REJECTS what is not on them. If
    # `_check_kwargs` were ever unwired, every assertion above would still pass
    # while an unsupported keyword went back to being silently absorbed by the
    # initializer — the exact failure `src/kwarg_validation.jl` exists to close.
    problem = DiagGaussian([0.0, 1.0], [1.0, 2.0])
    @test_throws ArgumentError adaptive_warmup_mcmc(
        Xoshiro(20260728), problem; n_draws=10, progress=nothing,
        definitely_not_a_warmuphmc_keyword=1)
end
