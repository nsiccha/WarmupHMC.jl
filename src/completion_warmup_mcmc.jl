# Completion is an orchestration policy over independent adaptive chains. It
# deliberately does not reuse the cooperative sampler's ESS/pool scheduler.
const _COMPLETION_SELECTION_WARNING = "Chains were selected by runtime completion. " *
    "Runtime can depend on the sampled states; retaining faster chains can bias inference. " *
    "R-hat, ESS and divergence checks on retained draws cannot rule out this selection bias."

# The caveat is a claim about an outcome: it applies when the retained chains are
# the ones that finished first, that is when some slot was cut short of `n_draws`
# (by the quorum, a callback or an interrupt) or never started while at least one
# chain was kept. A stuck or failed slot is a health verdict, not a race, and is
# reported in its own fields. Without such a slot the retained draws are exactly
# what all-chain sampling returns for the same seeds.
_runtime_selected(statuses) = any(==(:completed), statuses) &&
    any(st -> st in (:stopped, :not_started), statuses)

# Warn iff the outcome records the caveat. A terminal record written before the
# caveat depended on the outcome carries the text unconditionally, so reopening
# it still warns.
function _warn_runtime_selection(completion)
    text = get(completion, :selection_warning, "")
    isempty(text) || @warn text
    nothing
end

"An unmet completion quorum. `outcome` contains the settled results and every chain's disposition."
struct CompletionQuorumError{T} <: Exception
    outcome::T
end
Base.showerror(io::IO, e::CompletionQuorumError) = print(io,
    "Completion quorum unmet: ", e.outcome.completion.n_completed, " of ",
    e.outcome.completion.min_completed, " required chains completed",
    _stuck_note(e.outcome.completion), ". Inspect `outcome.completion.chains`.")
# A run recorded before the admission checks has no stuck chains to name.
_stuck_note(c) = isempty(get(c, :stuck_chain_indices, Int[])) ? "" :
    "; left out as stuck after reaching `n_draws`: $(c.stuck_chain_indices)"

# What a chain that reaches `n_draws` must also pass to count toward the quorum.
# Finishing first is no evidence of health: a chain whose step size collapsed, or
# whose transitions diverge, takes short cheap trajectories and can reach
# `n_draws` before every healthy sibling (snag quorum-admits-a-7aa7f768). A chain
# can also settle in a minor mode at an ordinary step size, without diverging
# (snag default-starts-t-013b07f7). Such a chain is left out as `:stuck`.
struct _CompletionAdmission
    # Left out when its step size is below the median of its siblings' step sizes
    # divided by this. `Inf` switches the check off.
    stuck_stepsize_ratio::Float64
    # Left out when more than this fraction of its retained draws diverged.
    # `1.0` switches the check off.
    max_admitted_divergent_fraction::Float64
    # Left out when its estimated log posterior mass (`_log_mass_estimate`) is
    # more than this many nats below the best sibling's, and more than
    # `_LOG_MASS_GAP_STANDARD_ERRORS` standard errors below it. `Inf` switches
    # the check off.
    max_admitted_log_mass_gap::Float64
end

# Healthy chains measured at most 2.8× below their siblings' median step size
# (4.4× when every chain collapsed together); the reported stuck chain sat 41×
# below, and both stuck chains measured were about 15% divergent (decision
# 1dl030w on WarmupHMC:snag.quorum-admits-a-7aa7f768). Healthy chains of a real
# BRM model reached 4.9% divergent draws at the default target acceptance
# (decision 1b2iq06 on WarmupHMC:snag.final-window-mos-3252ed56), hence 10%.
const _DEFAULT_STUCK_STEPSIZE_RATIO = 10.0
const _DEFAULT_MAX_ADMITTED_DIVERGENT_FRACTION = 0.1
# Measured on 136 healthy chains of 11 targets (Gaussians to d = 1000, AR(1) to
# d = 400, equal-mass mixtures of unequal widths, Student-t, centred funnels to
# d = 50, eight schools): the largest gap to the best sibling was 12.2 nats,
# except on the d = 50 funnel, whose 49.5-nat gap was within 2.3 standard
# errors. On a sinusoid regression with side lobes, the 3 lobe chains both other
# checks admitted sat about 167 nats below at standard errors near 0.1; the 7
# that wandered between lobes (standard errors 6 to 52) diverged and were left
# out by the divergence check (decision 1csfn2g on
# WarmupHMC:snag.default-starts-t-013b07f7).
const _DEFAULT_MAX_ADMITTED_LOG_MASS_GAP = 25.0
const _LOG_MASS_GAP_STANDARD_ERRORS = 4.0
# A chain's mass is estimated only from more than this many draws, the floor
# the per-coordinate ESS monitor uses too.
const _MIN_LOG_MASS_DRAWS = 10
# The scheduling seams below default to admitting every full result, so the
# controlled-timing tests that do not exercise admission need no step sizes.
const _ADMIT_EVERY_FULL_CHAIN = _CompletionAdmission(Inf, 1.0, Inf)

_validate_stuck_stepsize_ratio(r) = begin
    r isa Real && !(r isa Bool) && r >= 1 || throw(ArgumentError(
        "`stuck_stepsize_ratio` must be a number ≥ 1 (Inf switches the step-size " *
        "check off); got $(repr(r))"))
    Float64(r)
end
_validate_max_admitted_divergent_fraction(f) = begin
    f isa Real && !(f isa Bool) && 0 <= f <= 1 || throw(ArgumentError(
        "`max_admitted_divergent_fraction` must be a number in [0, 1] (1 switches " *
        "the divergence check off); got $(repr(f))"))
    Float64(f)
end
_validate_max_admitted_log_mass_gap(g) = begin
    g isa Real && !(g isa Bool) && g >= 0 || throw(ArgumentError(
        "`max_admitted_log_mass_gap` must be a number ≥ 0, in nats (Inf switches " *
        "the mass check off); got $(repr(g))"))
    Float64(g)
end
_mass_check(a::_CompletionAdmission) = isfinite(a.max_admitted_log_mass_gap)

# The log posterior mass near a chain's draws, up to a constant shared by every
# chain of one target: the mean log density of the draws plus the log of the
# volume they spread over, `Σᵢ log sdᵢ` of their per-coordinate standard
# deviations. For a Gaussian mode this is its log mass up to that constant and
# to ignoring the draws' correlations, which only enlarges the volume. Not the
# mean log density alone: a wide genuine mode has a lower one than a narrow mode
# of equal mass. `se` is the Monte Carlo standard error of the mean log density,
# `sd / √ESS`. Linear in the dimension; no covariance is formed.
#
# `positions` hold the draws in a sampling frame and `log_density` the density
# `ℓq` there; `ir` maps them into the model frame (`nothing` when they are in it
# already), which every chain of one target shares, whatever centering each
# adapted. `nothing` with at most `_MIN_LOG_MASS_DRAWS` draws.
function _log_mass_estimate(ir, positions, log_density)
    d, n = size(positions)
    length(log_density) == n || throw(DimensionMismatch(
        "one log density per draw is required; got $(length(log_density)) for $n draws"))
    n > _MIN_LOG_MASS_DRAWS || return nothing
    mapped = !isnothing(ir) && has_reparametrization(ir)
    lp = Vector{Float64}(undef, n)
    # Welford's per-coordinate moments, so no model-frame copy of the draws is made.
    mu, m2 = zeros(d), zeros(d)
    for j in 1:n
        x = view(positions, :, j)
        ljac, y = mapped ? ir(x) : (0.0, x)
        lp[j] = log_density[j] - ljac
        for k in 1:d
            delta = y[k] - mu[k]
            mu[k] += delta / j
            m2[k] += delta * (y[k] - mu[k])
        end
    end
    log_volume = sum(k -> log(m2[k] / (n - 1)) / 2, 1:d)
    sd = std(lp)
    ess = MCMCDiagnosticTools.ess(reshape(lp, n, 1, 1))[1]
    # A non-finite ESS (a constant log density, say) is counted as one draw.
    se = iszero(sd) ? 0.0 : sd / sqrt(isfinite(ess) && ess > 0 ? ess : 1.0)
    (; log_mass=mean(lp) + log_volume, se)
end
# A returned result is in the model frame already. A scheduling-test fixture
# without log densities has no estimate.
_result_log_mass(r) = hasproperty(r, :posterior_log_density) ?
    _log_mass_estimate(nothing, r.posterior_position, r.posterior_log_density) : nothing
_state_log_mass(state) = _log_mass_estimate(reparametrizer(state.lpdf),
    state.recording_lpdf.posterior_position, state.posterior_log_density)
# A partial checkpoint's draws are in its own sampling frame; its saved
# centering maps them. A checkpoint written before the log densities were
# recorded has no estimate until the chain resumes and reaches a boundary.
_payload_log_mass(payload, lpdf) = haskey(payload, :posterior_log_density) ?
    _log_mass_estimate(_back_transform_reparametrizer(payload, lpdf, payload.dimension),
        payload.posterior_position, payload.posterior_log_density) : nothing

# The step size before any opt-in halving (`divergence_tolerance`,
# `_halve_stepsize!`): what the stuck check compares. A halving answers
# divergences, which the divergence check judges; it is not a collapsed
# adaptation, and a chain that halved while its siblings did not must not look
# stuck for it. Exact until a restart re-adapts for the recorded effective
# target acceptance; after one, the leapfrog-scaling approximation the halving
# itself assumes. `x` is a sampler state, a result or a checkpoint payload.
_adapted_stepsize(x) = ldexp(Float64(x.stepsize),
    hasproperty(x, :stepsize_halvings) ? x.stepsize_halvings : 0)

# The admission record of a slot that never reached `n_draws`.
const _UNJUDGED = (; stuck_reason=nothing, stepsize=nothing, reference_stepsize=nothing,
    divergent_fraction=nothing, log_mass=nothing, log_mass_se=nothing,
    reference_log_mass=nothing, reference_log_mass_se=nothing)

mutable struct _CompletionState{C}
    lock::ReentrantLock
    clock::C
    start::Float64
    min_completed::Int
    admission::_CompletionAdmission
    quorum_at::Union{Nothing,Float64}
    statuses::Vector{Symbol}
    results::Vector{Any}
    errors::Vector{Any}
    finished_at::Vector{Union{Nothing,Float64}}
    # Each slot's adapted step size (`_adapted_stepsize`) at its latest checkpoint
    # boundary, or `nothing` while it
    # keeps no draws there (still initializing, or its window just restarted).
    sampling_stepsizes::Vector{Union{Nothing,Float64}}
    # Each slot's `_log_mass_estimate` at its latest boundary, or `nothing` while
    # it has none (too few draws kept there, or the mass check is off).
    log_masses::Vector{Any}
    # Each judged slot's admission record; `_UNJUDGED` until it reaches `n_draws`.
    admissions::Vector{Any}
end

_CompletionState(n; clock, min_completed, admission=_ADMIT_EVERY_FULL_CHAIN) =
    _CompletionState(ReentrantLock(), clock, Float64(clock()), min_completed, admission,
        nothing, fill(:pending, n), Any[nothing for _ in 1:n], Any[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n], Any[nothing for _ in 1:n],
        Any[_UNJUDGED for _ in 1:n])

_completion_elapsed(s::_CompletionState) = Float64(s.clock()) - s.start
function _completion_stop(s::_CompletionState)
    lock(s.lock) do
        !isnothing(s.quorum_at)
    end
end

# What a running chain sees of the batch. Calling it asks whether the quorum has
# requested a stop; `_report_boundary!` publishes the step size it samples with
# and its mass estimate, which a sibling's step-size and mass checks compare against.
struct _CompletionControl{S,P}
    state::S
    slot::Int
    progress::P
end
(c::_CompletionControl)() = _completion_stop(c.state)

function _report_boundary!(c::_CompletionControl, stepsize, n_samples, log_mass=nothing)
    s = c.state
    changed = lock(s.lock) do
        before = copy(s.statuses)
        s.sampling_stepsizes[c.slot] = n_samples > 0 ? Float64(stepsize) : nothing
        s.log_masses[c.slot] = log_mass
        _settle_admissions!(s, _completion_elapsed(s))
        s.statuses != before
    end
    changed && _update_completion_progress!(c.progress, s)
    nothing
end

# Must hold `s.lock`. Judge slot `i`, whose result holds its full `n_draws`.
# Divergences first, as they need no sibling; then the step size and the mass,
# against the siblings', which may mean waiting for a sibling to keep draws.
function _admit!(s::_CompletionState, i)
    r = s.results[i]
    own = s.log_masses[i]
    s.admissions[i] = (; stuck_reason=nothing, stepsize=_adapted_stepsize(r),
        reference_stepsize=nothing,
        divergent_fraction=r.n_divergent_samples / size(r.posterior_position, 2),
        log_mass=isnothing(own) ? nothing : own.log_mass,
        log_mass_se=isnothing(own) ? nothing : own.se,
        reference_log_mass=nothing, reference_log_mass_se=nothing)
    if s.admissions[i].divergent_fraction > s.admission.max_admitted_divergent_fraction
        _leave_out!(s, i, :divergence)
    elseif isinf(s.admission.stuck_stepsize_ratio) && !_mass_check(s.admission)
        s.statuses[i] = :completed
    else
        s.statuses[i] = :awaiting_reference
        _judge_against_siblings!(s, i)
    end
    nothing
end

# Must hold `s.lock`. Compare awaiting slot `i` with its siblings that keep
# draws: its step size with their median, then its mass with the largest of
# theirs. Without a sibling to compare with it keeps waiting, uncounted, while
# one may still start sampling; once none can, the comparison is skipped. Each
# comparison is made once, so a verdict is final and the quorum count only ever
# grows. A chain without a mass estimate of its own (at most
# `_MIN_LOG_MASS_DRAWS` draws) is not compared on mass.
function _judge_against_siblings!(s::_CompletionState, i)
    can_wait = any(k -> k != i && s.statuses[k] in (:pending, :running), eachindex(s.statuses))
    a = s.admissions[i]
    if !isinf(s.admission.stuck_stepsize_ratio) && isnothing(a.reference_stepsize)
        reference = _sibling_stepsize(s, i)
        if isnothing(reference)
            can_wait && return nothing
        else
            s.admissions[i] = a = merge(a, (; reference_stepsize=reference))
            a.stepsize < reference / s.admission.stuck_stepsize_ratio &&
                return _leave_out!(s, i, :stepsize)
        end
    end
    if _mass_check(s.admission) && !isnothing(a.log_mass)
        reference = _sibling_log_mass(s, i)
        if isnothing(reference)
            can_wait && return nothing
        else
            s.admissions[i] = a = merge(a, (; reference_log_mass=reference.log_mass,
                reference_log_mass_se=reference.se))
            _low_mass(a, s.admission) && return _leave_out!(s, i, :low_mass)
        end
    end
    s.statuses[i] = :completed
    nothing
end

# Both bounds must be exceeded: the gap in nats, and the gap in combined
# standard errors, which keeps a noisy estimate such as a funnel's.
_low_mass(a, admission) = begin
    gap = a.reference_log_mass - a.log_mass
    gap > admission.max_admitted_log_mass_gap &&
        gap > _LOG_MASS_GAP_STANDARD_ERRORS * hypot(a.log_mass_se, a.reference_log_mass_se)
end

function _leave_out!(s::_CompletionState, i, reason)
    s.admissions[i] = merge(s.admissions[i], (; stuck_reason=reason))
    s.statuses[i] = :stuck
    nothing
end

# A stuck or failed sibling is no reference for a healthy step size.
function _sibling_stepsize(s::_CompletionState, i)
    steps = Float64[s.sampling_stepsizes[k] for k in eachindex(s.statuses)
        if k != i && !isnothing(s.sampling_stepsizes[k]) && s.statuses[k] ∉ (:stuck, :failed)]
    isempty(steps) ? nothing : median(steps)
end

# Nor for a mass. The reference is the largest: a chain is compared with where
# its siblings found the most mass, not with their typical place.
function _sibling_log_mass(s::_CompletionState, i)
    best = nothing
    for k in eachindex(s.statuses)
        k != i && s.statuses[k] ∉ (:stuck, :failed) || continue
        estimate = s.log_masses[k]
        isnothing(estimate) && continue
        (isnothing(best) || estimate.log_mass > best.log_mass) && (best = estimate)
    end
    best
end

# Must hold `s.lock`. Re-judge every awaiting slot after any change a judgement
# depends on, then record when the quorum is first met.
function _settle_admissions!(s::_CompletionState, now)
    for i in eachindex(s.statuses)
        s.statuses[i] === :awaiting_reference && _judge_against_siblings!(s, i)
    end
    if isnothing(s.quorum_at) && count(==(:completed), s.statuses) >= s.min_completed
        s.quorum_at = now
    end
    nothing
end

function _completion_worker!(run_chain, s::_CompletionState, i; progress=nothing)
    started = lock(s.lock) do
        if s.statuses[i] in (:completed, :stuck, :awaiting_reference)
            # Already judged by the resume pre-pass.
            false
        elseif !isnothing(s.quorum_at) || _interrupted(progress)
            # Quorum met, or a stop requested through the progress tree before
            # this slot started: never pay a fresh chain's initialization.
            s.statuses[i] = :not_started
            _settle_admissions!(s, _completion_elapsed(s))
            false
        else
            s.statuses[i] = :running
            true
        end
    end
    if started
        try
            result, complete = run_chain(i, _CompletionControl(s, i, progress))
            lock(s.lock) do
                now = _completion_elapsed(s)
                s.results[i] = result
                s.finished_at[i] = now
                s.sampling_stepsizes[i] = size(result.posterior_position, 2) > 0 ?
                    _adapted_stepsize(result) : nothing
                s.log_masses[i] = _result_log_mass(result)
                # Quorum requests a stop at the next checkpoint; it never closes
                # admission. A full result from the final round is still judged.
                complete ? _admit!(s, i) : (s.statuses[i] = :stopped)
                _settle_admissions!(s, now)
            end
        catch err
            # A failed chain cannot strand a worker or masquerade as a completion.
            # Preserve the exception AND backtrace, then join the other workers.
            failure = CapturedException(err, catch_backtrace())
            lock(s.lock) do
                now = _completion_elapsed(s)
                s.errors[i] = failure
                s.finished_at[i] = now
                s.statuses[i] = :failed
                _settle_admissions!(s, now)
            end
        end
    end
    # Every exit settles a slot — including quorum-already-met `:not_started`
    # markings, which never reach `run_chain`.
    _update_completion_progress!(progress, s)
    nothing
end

_completion_slots_text(slots) = isempty(slots) ? "none" : string(slots)

# Publish one live quorum snapshot to the parent node: the counter advances per
# completed chain against the quorum target, and the labels carry the exact
# finished count (the rendered counter clamps at N, so over-quorum completions
# would otherwise be invisible), the stop state, and every non-completed slot.
# Concurrent settlers each publish their own snapshot; last-writer-wins races
# are harmless because the caller re-publishes the settled truth afterwards.
function _update_completion_progress!(progress, s::_CompletionState)
    isnothing(progress) && return nothing
    snap = lock(s.lock) do
        (; n_completed=count(==(:completed), s.statuses),
           stop_requested=!isnothing(s.quorum_at),
           stopped=findall(==(:stopped), s.statuses),
           not_started=findall(==(:not_started), s.statuses),
           failed=findall(==(:failed), s.statuses),
           stuck=findall(==(:stuck), s.statuses),
           awaiting_reference=findall(==(:awaiting_reference), s.statuses))
    end
    update_progress!(progress, snap.n_completed;
        finished="$(snap.n_completed) of $(s.min_completed)",
        stop_requested=string(snap.stop_requested),
        stopped=_completion_slots_text(snap.stopped),
        not_started=_completion_slots_text(snap.not_started),
        failed=_completion_slots_text(snap.failed),
        stuck=_completion_slots_text(snap.stuck),
        awaiting_reference=_completion_slots_text(snap.awaiting_reference))
    nothing
end

# Settle the quorum parent from the joined outcome: absolute counter and labels
# (correcting any last-writer-wins race between concurrent settlers), then the
# final message naming the stop reason, the finished count, and the pooled
# divergence rate over retained draws. Runs before the quorum throw, so a failed
# run's tree still shows the outcome. No pooled ESS: retained chains can carry
# different draw counts, so their positions do not stack.
function _finish_completion_progress!(cprogress, completion)
    isnothing(cprogress) && return nothing
    c = completion
    stopped = [ch.chain_index for ch in c.chains if ch.status === :stopped]
    not_started = [ch.chain_index for ch in c.chains if ch.status === :not_started]
    update_progress!(cprogress, c.n_completed;
        finished="$(c.n_completed) of $(c.min_completed)",
        stop_requested=string(!isnothing(c.quorum_at_seconds)),
        stopped=_completion_slots_text(stopped),
        not_started=_completion_slots_text(not_started),
        failed=_completion_slots_text(c.failed_chain_indices),
        stuck=_completion_slots_text(c.stuck_chain_indices),
        # Every awaiting slot is judged before the batch returns.
        awaiting_reference=_completion_slots_text(Int[]))
    divergent = c.n_samples > 0 ?
        ", divergent: $(short_string(100 * c.n_divergent_samples / c.n_samples))%" : ""
    update_progress!(cprogress,
        "$(c.stop_reason): $(c.n_completed) of $(c.min_completed) chains completed$(divergent)")
    nothing
end

# The injected runner/clock are internal seams for controlled scheduling tests;
# the public method below always runs the real adaptive sampler and monotonic time.
function _completion_batch(run_chain, n; min_completed, admission=_ADMIT_EVERY_FULL_CHAIN,
        initial_results=Dict{Int,Any}(), initial_stepsizes=Dict{Int,Float64}(),
        initial_log_masses=Dict{Int,Any}(), clock=() -> time_ns() / 1e9, progress=nothing)
    s = _CompletionState(n; clock, min_completed, admission)
    # Resume admission is a deterministic pre-pass, not a race between tasks
    # restoring already-complete checkpoints. Every original slot's latest
    # checkpointed step size and mass estimate are in place first, so each
    # already-complete slot is judged against all of them, as a live chain is
    # against its siblings.
    for (i, stepsize) in initial_stepsizes
        s.sampling_stepsizes[i] = stepsize
    end
    for (i, estimate) in initial_log_masses
        s.log_masses[i] = estimate
    end
    for (i, result) in initial_results
        s.results[i] = result
        s.finished_at[i] = 0.0
        s.sampling_stepsizes[i] = _adapted_stepsize(result)
        s.log_masses[i] = _result_log_mass(result)
    end
    foreach(i -> _admit!(s, i), sort!(collect(keys(initial_results))))
    _settle_admissions!(s, 0.0)
    _update_completion_progress!(progress, s)
    @sync for i in 1:n
        Threads.@spawn _completion_worker!(run_chain, s, i; progress)
    end
    # All workers are joined before any result or terminal file is published.
    # With no slot left running, every awaiting slot is judged here at the latest.
    lock(() -> _settle_admissions!(s, _completion_elapsed(s)), s.lock)
    completed = findall(==(:completed), s.statuses)
    failed = findall(==(:failed), s.statuses)
    stuck = findall(==(:stuck), s.statuses)
    started = findall(!=(:not_started), s.statuses)
    omitted = setdiff(collect(1:n), completed)
    results = [merge(s.results[i], (; chain_index=i,
        n_samples=size(s.results[i].posterior_position, 2))) for i in completed]
    chains = map(1:n) do i
        r = s.results[i]
        (; chain_index=i, status=s.statuses[i], finished_at_seconds=s.finished_at[i],
           # A failed worker may have checkpointed draws before throwing; no
           # finalized result exists from which to claim its sampled count.
           n_samples=s.statuses[i] === :failed ? nothing : isnothing(r) ? 0 : size(r.posterior_position, 2),
           n_retained_samples=i in completed ? size(r.posterior_position, 2) : 0,
           error=s.errors[i], s.admissions[i]...)
    end
    n_samples = sum(r -> r.n_samples, results; init=0)
    n_divergent_samples = sum(r -> r.n_divergent_samples, results; init=0)
    quorum_met = length(completed) >= min_completed
    reason = !quorum_met ? :quorum_unmet : isempty(omitted) ? :all_completed : :quorum_reached
    completion = (; schema_version=4, policy=:completion_quorum,
        stop_policy=:finish_current_round, min_completed,
        admission.stuck_stepsize_ratio, admission.max_admitted_divergent_fraction,
        admission.max_admitted_log_mass_gap,
        n_requested=n, n_started=length(started), n_completed=length(completed),
        n_failed=length(failed), n_stuck=length(stuck), n_omitted=length(omitted),
        requested_chain_indices=collect(1:n), started_chain_indices=started,
        completed_chain_indices=completed, failed_chain_indices=failed,
        stuck_chain_indices=stuck,
        omitted_chain_indices=omitted, chains, quorum_met, stop_reason=reason,
        quorum_at_seconds=s.quorum_at, stop_requested_at_seconds=s.quorum_at,
        elapsed_seconds=_completion_elapsed(s), n_samples, n_divergent_samples,
        selection_warning=_runtime_selected(s.statuses) ? _COMPLETION_SELECTION_WARNING : "")
    (; results, completion)
end

function _completion_directory(dir, n, resume, overwrite)
    resume && overwrite && throw(ArgumentError("`resume` and `overwrite` are mutually exclusive."))
    if isnothing(dir)
        (resume || overwrite) && throw(ArgumentError("`resume`/`overwrite` require `checkpoint_dir`."))
        return nothing
    end
    manifest_path = joinpath(dir, "completion_manifest.jls")
    if resume
        isfile(manifest_path) || throw(ArgumentError(
            "No completion-policy manifest in $(repr(dir)); cannot infer original chain identities."))
        manifest = deserialize(manifest_path)
        manifest.schema_version in (1, 2) && manifest.sampler === :completion ||
            throw(ArgumentError("Unsupported completion-policy manifest."))
        manifest.n_requested == n || throw(ArgumentError(
            "Resume requires the original $(manifest.n_requested) RNG slots, in their original order."))
    else
        # Reuse the existing whole-run guard, including its explicit overwrite
        # semantics. completion_* entries are ours and guarded separately.
        existing = isdir(dir) ? filter(f -> startswith(f, "completion_"), readdir(dir)) : String[]
        !isempty(existing) && !overwrite && throw(ArgumentError(
            "Completion run already exists in $(repr(dir)); pass `resume=true` or `overwrite=true`."))
        guard_run_dir!(dir, false, overwrite, :completion)
        overwrite && foreach(f -> rm(joinpath(dir, f); recursive=true), existing)
        _atomic_serialize_manifest(manifest_path, (; schema_version=2, sampler=:completion, n_requested=n))
    end
    # Each invocation has its own write-once manifest and terminal summary, so
    # a previous invocation's summary never signals completion of a resume.
    attempts = joinpath(dir, "completion_attempts")
    mkpath(attempts)
    attempt = 1
    while ispath(joinpath(attempts, "attempt_$attempt"))
        attempt += 1
    end
    path = joinpath(attempts, "attempt_$attempt")
    mkdir(path)
    path
end

function _atomic_serialize_manifest(path, payload)
    mkpath(dirname(path))
    _atomic_serialize(path, payload)
end

function _write_completion_summary(dir, completion)
    isnothing(dir) && return
    # Serialize exceptions as human-readable text in the portable JSON view.
    fields = Pair{String,Any}[string(k) => v for (k, v) in pairs(completion) if k !== :chains]
    push!(fields, "chain_statuses" => string.(getproperty.(completion.chains, :status)))
    push!(fields, "chain_finished_at_seconds" => getproperty.(completion.chains, :finished_at_seconds))
    push!(fields, "chain_n_samples" => getproperty.(completion.chains, :n_samples))
    push!(fields, "chain_n_retained_samples" => getproperty.(completion.chains, :n_retained_samples))
    push!(fields, "chain_errors" => [isnothing(c.error) ? nothing : sprint(showerror, c.error)
        for c in completion.chains])
    # A terminal record from before the admission checks, or from before the
    # mass check, has no such fields.
    if haskey(completion, :stuck_chain_indices)
        for (name, key) in ("chain_stuck_reasons" => :stuck_reason, "chain_stepsizes" => :stepsize,
                "chain_reference_stepsizes" => :reference_stepsize,
                "chain_divergent_fractions" => :divergent_fraction)
            push!(fields, name => getproperty.(completion.chains, key))
        end
    end
    if haskey(completion, :max_admitted_log_mass_gap)
        for (name, key) in ("chain_log_masses" => :log_mass, "chain_log_mass_ses" => :log_mass_se,
                "chain_reference_log_masses" => :reference_log_mass,
                "chain_reference_log_mass_ses" => :reference_log_mass_se)
            push!(fields, name => getproperty.(completion.chains, key))
        end
    end
    _write_json(joinpath(dir, "run_summary.json"), fields)
end

"""
    completion_warmup_mcmc(rngs, lpdf_or_lpdfs; min_completed=length(rngs),
        n_draws=1000, stuck_stepsize_ratio=10, max_admitted_divergent_fraction=0.1,
        max_admitted_log_mass_gap=25, checkpoint_dir=nothing, resume=false,
        overwrite=false, kwargs...)

Opt-in completion quorum over independent [`adaptive_warmup_mcmc`](@ref)
chains. For example, pass 16 RNGs and `min_completed=12` to request that
unfinished chains finish their current round when the twelfth full chain
returns. Every chain completing its requested production draws during that
final round is retained too, in original index order, unless it is stuck (below).
The call returns once all workers settle. The default samplers and their return
types are unchanged.

**Stuck chains are left out.** Reaching `n_draws` first is no evidence of health:
a chain whose step size collapsed, or whose transitions diverge, takes short,
cheap trajectories and can finish before every healthy chain. A chain can also
settle in a minor mode, far below its siblings, at an ordinary step size and
without diverging. A chain that reaches `n_draws` counts toward `min_completed`
only if

- at most `max_admitted_divergent_fraction` of its retained draws diverged,
- its step size is at least `1/stuck_stepsize_ratio` of the median latest step
  size of its siblings that are keeping draws at that point. Both are the step
  sizes adaptation chose, before any opt-in halving (`divergence_tolerance`): a
  halving answers divergences, which the first check judges; and
- its estimated log posterior mass is not more than `max_admitted_log_mass_gap`
  nats below the largest of its siblings' estimates at that point, or not more
  than 4 standard errors below it. The estimate is the mean model-frame log
  density of the retained draws plus `Σᵢ log sdᵢ`, the log of the volume their
  per-coordinate standard deviations span, so a wide mode is not mistaken for
  a minor one for its lower density; the standard error is that of the mean log
  density (`sd / √ESS`). Each chain is compared in the model's own
  parametrization, whatever centering it adapted. Chains with at most 10 draws
  are not compared.

Otherwise its status is `:stuck`: it is omitted from `results` and does not
count, so the quorum waits for chains that pass. While no sibling keeps draws
yet, a finished chain waits uncounted and is judged as soon as one does; if
none ever can, it is admitted. A verdict is final once made. The mass check
compares chains only with each other: when every chain sits in one minor mode
it keeps them all, and its volume ignores correlations, so two genuine modes
of equal mass that differ mainly in their correlations are compared by their
marginal spreads. The step-size check cannot catch a run in which every chain
collapsed together.
`stuck_stepsize_ratio=Inf`, `max_admitted_divergent_fraction=1` and
`max_admitted_log_mass_gap=Inf` switch the checks off.

Each chain runs the ordinary adaptive procedure with its own deep-copied
density, initialization and RNG slot. There is no cross-chain adaptation.
`init` accepts the adaptive multi-chain per-chain forms. Other `kwargs` are
adaptive single-chain keywords, including `callback(state, stage)`: returning
`true` there stops only that chain and does not count an incomplete chain.
`Treebars.request_interrupt!` behaves the same way through the progress tree: on
one chain's node it stops that chain, on the node passed as `progress` (or an
ancestor) it stops every running chain and starts no further slot.
Callbacks may run concurrently and must be observational and thread-safe.
`blas_threads=1` holds BLAS at that many threads for the whole call, every chain
included, as for [`adaptive_warmup_mcmc`](@ref); `nothing` leaves BLAS
untouched.
`progress=` builds one quorum parent node — chains completed of `min_completed`,
the stop state, and the stopped/never-started/failed slots — with one
`"<description>.<i>"` child subtree per original chain slot, as multi-chain
adaptive. A worker that never starts leaves no child; its slot is named by the
parent's labels instead.

Quorum requests a cooperative stop at the next checkpoint boundary. An active
initialization or adaptive window must finish; this is **not a hard wall-time
limit**. A worker that has not started at quorum does not start. No tasks are
interrupted or detached, and every worker is joined before return. There is no
grace period or admission cutoff: a full result is always retained.
Julia threads are required for simultaneous CPU work; runtime selection and
completion order are not reproducible from RNG seeds alone.

Returns `(; results, completion)`. Only admitted full chains appear in
`results`, each carrying the adaptive result fields plus original `chain_index`
and actual `n_samples`. `completion` records policy and the three admission
thresholds, requested/started/completed/failed/stuck/omitted identities, all
chain dispositions and captured errors, quorum and stop-request times (seconds
since worker scheduling begins), and retained draw/divergence counts. Each
`completion.chains[i]` of a chain that reached `n_draws` also records its
adapted `stepsize` (before any halving; the result's own `stepsize` is the one
it sampled with), `divergent_fraction`, the `reference_stepsize` it was compared with
(`nothing` when it was not compared: no sibling ever kept draws, the step-size
check is off, or the divergence check already left it out), its `log_mass` and
`log_mass_se` (`nothing` with at most 10 draws), the `reference_log_mass` and
`reference_log_mass_se` it was compared with (`nothing` when it was not
compared), and, for a stuck chain, `stuck_reason` (`:divergence`, `:stepsize`
or `:low_mass`). A failed chain's `n_samples` is `nothing` because it
has no finalized result; its `n_retained_samples` is zero. Failed and stuck
chains are also omitted. If the quorum is unmet, throws
`WarmupHMC.CompletionQuorumError` with this same `outcome`; partial results are
never silently reported as a successful fit.

**Inferential limitation:** selecting faster chains can bias inference when
runtime depends on sampled states or modes. R-hat, bulk/tail ESS and divergence
checks describe only the returned draws and cannot rule out that bias. The call
`@warn`s this caveat after the chains settle, and records it in
`completion.selection_warning`, when the retained chains are such a selection:
at least one chain was kept and some other slot was stopped short of `n_draws`
(by the quorum, a callback or an interrupt) or never started. Otherwise
`selection_warning` is `""` and nothing is printed: with
`min_completed = length(rngs)`, or when every chain finished anyway, the returned
draws are those of all-chain sampling. Chains left out as stuck or failed do not
make a selection; they are named in `stuck_chain_indices` and
`failed_chain_indices`. Report the policy and all omissions; do not describe the
requested count as completed. Use all-chain sampling when this selection is
unacceptable.

Checkpoint files remain ordinary adaptive source-frame state under original
`chain_<i>` directories. `resume=true` requires the same ordered RNG/density
slots; existing checkpoints restore their saved RNG, while a slot that never
checkpointed initializes from the supplied RNG. Re-pass original sampler
settings. For an interrupted run, already-complete checkpoints are judged
deterministically before workers launch, against every slot's latest
checkpointed step size and mass estimate; no incomplete worker starts if the
admitted ones already meet quorum. A partial checkpoint written before the samplers
recorded log densities supplies no mass estimate until its chain resumes and
reaches a boundary. A successful terminal run reopens its **recorded outcome**, and
warns iff its recorded `selection_warning` is non-empty (always, for runs
recorded before the caveat followed the outcome):
it never samples or admits omitted chains, and changing `n_draws`,
`min_completed`, the admission thresholds or the slot count is rejected. A run
recorded before the admission checks existed reopens as recorded, whatever
thresholds are passed, and one recorded before the mass check whatever
`max_admitted_log_mass_gap` is passed. This includes terminal runs
saved under the older grace/cutoff policy: their recorded selection and
metadata are preserved, not reinterpreted. Omit the removed `grace_seconds`
keyword when reopening them. An unfinished legacy run resumes using the new
finish-current-round policy. Extend individual adaptive
checkpoints separately when a different production target is required.
`completion_manifest.jls` binds
the original slot count; each invocation writes its own `run_manifest.json`
(whose `selection_warning` carries the caveat when `min_completed` is below the
slot count, `""` otherwise) and, after all workers settle, `run_summary.json` under
`completion_attempts/attempt_<n>/`. The returned `completion.attempt_directory`
identifies that invocation. `completion_terminal.jls` stores the successful
outcome, including returned draws, atomically before the summary is published.
These run records do not change chain payloads.
"""
completion_warmup_mcmc(rngs::AbstractVector, lpdfs::AbstractVector;
        min_completed=length(rngs), n_draws=1000,
        stuck_stepsize_ratio=_DEFAULT_STUCK_STEPSIZE_RATIO,
        max_admitted_divergent_fraction=_DEFAULT_MAX_ADMITTED_DIVERGENT_FRACTION,
        max_admitted_log_mass_gap=_DEFAULT_MAX_ADMITTED_LOG_MASS_GAP,
        init=missing, checkpoint_dir=nothing, resume=false, overwrite=false,
        callback=nothing, progress=nothing, description="MCMC",
        monitor_ess=!isnothing(progress), blas_threads=_DEFAULT_BLAS_THREADS,
        kwargs...) = _with_blas_threads(blas_threads) do
    n = length(rngs)
    Base.require_one_based_indexing(rngs, lpdfs)
    n > 0 || throw(ArgumentError("At least one RNG is required."))
    length(lpdfs) == n || throw(DimensionMismatch("One density per RNG is required."))
    min_completed isa Integer && !(min_completed isa Bool) && 1 <= min_completed <= n ||
        throw(ArgumentError("`min_completed` must be an integer in 1:$n."))
    n_draws isa Integer && !(n_draws isa Bool) && 0 < n_draws < typemax(Int) ||
        throw(ArgumentError("`n_draws` must be a finite positive integer."))
    admission = _CompletionAdmission(_validate_stuck_stepsize_ratio(stuck_stepsize_ratio),
        _validate_max_admitted_divergent_fraction(max_admitted_divergent_fraction),
        _validate_max_admitted_log_mass_gap(max_admitted_log_mass_gap))
    haskey(kwargs, :grace_seconds) && throw(ArgumentError(
        "`grace_seconds` was removed. Quorum now requests a stop after each running chain's " *
        "current round, and every full result is retained. Omit this keyword, including on resume."))
    _check_kwargs(:completion_warmup_mcmc, kwargs)
    inits = ensurevector(init, n)
    resume && overwrite && throw(ArgumentError("`resume` and `overwrite` are mutually exclusive."))
    terminal_path = isnothing(checkpoint_dir) ? nothing : joinpath(checkpoint_dir, "completion_terminal.jls")
    if resume && !isnothing(terminal_path) && isfile(terminal_path)
        saved = deserialize(terminal_path)
        saved.schema_version in (1, 2) || throw(ArgumentError("Unsupported terminal completion schema."))
        c = saved.outcome.completion
        # A run recorded before admission checks existed carries no thresholds;
        # its selection is reopened as recorded, whatever thresholds are passed.
        # Likewise the mass threshold for a run recorded before the mass check.
        same_admission = !haskey(c, :stuck_stepsize_ratio) ||
            (c.stuck_stepsize_ratio == admission.stuck_stepsize_ratio &&
             c.max_admitted_divergent_fraction == admission.max_admitted_divergent_fraction &&
             (!haskey(c, :max_admitted_log_mass_gap) ||
              c.max_admitted_log_mass_gap == admission.max_admitted_log_mass_gap))
        (c.n_requested == n && saved.n_draws == n_draws &&
            c.min_completed == min_completed && same_admission) ||
            throw(ArgumentError("A finished completion run has an immutable target, policy and chain selection. " *
                "Reopen with its original configuration or use a new directory."))
        # Recover a crash after the atomic terminal write but before its JSON
        # publication, without reopening the selection decision.
        isfile(joinpath(c.attempt_directory, "run_summary.json")) ||
            _write_completion_summary(c.attempt_directory, c)
        _warn_runtime_selection(c)
        return saved.outcome
    end
    # Every slot gets a task; the default thread pool runs them.
    _refuse_unsafe_concurrent_gradients(:completion_warmup_mcmc, lpdfs, _concurrent_chains(n, n))
    # Copies are made serially, before any worker can mutate its density.
    densities = map(deepcopy, lpdfs)
    attempt_directory = _completion_directory(checkpoint_dir, n, resume, overwrite)
    if !isnothing(attempt_directory)
        _write_json(joinpath(attempt_directory, "run_manifest.json"), Pair{String,Any}[
            "schema_version" => 4, "sampler" => "completion", "n_requested" => n,
            "min_completed" => min_completed, "stop_policy" => "finish_current_round",
            # A switched-off step-size or mass check (`Inf`) is written as `null`.
            "stuck_stepsize_ratio" => admission.stuck_stepsize_ratio,
            "max_admitted_divergent_fraction" => admission.max_admitted_divergent_fraction,
            "max_admitted_log_mass_gap" => admission.max_admitted_log_mass_gap,
            "n_draws" => n_draws, "resume" => resume,
            # Written before any chain runs, so it states the policy: only a
            # quorum below the slot count can leave a chain out by runtime.
            "selection_warning" => min_completed < n ? _COMPLETION_SELECTION_WARNING : ""])
    end
    # One quorum parent over the whole batch, as multi-chain adaptive: the
    # counter fills at `min_completed`, each chain hangs its ordinary adaptive
    # subtree under it labelled by ORIGINAL slot index.
    outcome = with_progress(progress, Int(min_completed); description) do cprogress
        run_chain = function (i, control)
            dir = _chain_dir(checkpoint_dir, i)
            resume_chain = resume && !isempty(_checkpoint_files(dir))
            boundary = (state, stage) -> begin
                user_stop = _fire_callback(callback, state, stage)
                # Published before the stop query: it can settle a sibling that
                # waits for a step size or a mass to be compared with, and so
                # the quorum. Read-only, so the chain samples as it would alone.
                _report_boundary!(control, _adapted_stepsize(state),
                    size(state.recording_lpdf.posterior_position, 2),
                    _mass_check(admission) ? _state_log_mass(state) : nothing)
                control() || user_stop
            end
            result = adaptive_warmup_mcmc(rngs[i], densities[i]; n_draws, init=deepcopy(inits[i]),
                checkpoint_dir=dir, resume=resume_chain, callback=boundary,
                progress=cprogress, description=description * ".$i", monitor_ess, blas_threads,
                kwargs...)
            result, size(result.posterior_position, 2) >= n_draws
        end
        initial_results = Dict{Int,Any}()
        initial_stepsizes = Dict{Int,Float64}()
        initial_log_masses = Dict{Int,Any}()
        if resume
            for i in 1:n
                latest = joinpath(_chain_dir(checkpoint_dir, i), "cp_latest.jls")
                isfile(latest) || continue
                payload = deserialize(latest)
                retained = size(payload.posterior_position, 2)
                if retained >= n_draws
                    # A complete restore runs no window, so it reaches no boundary.
                    result, complete = run_chain(i, () -> false)
                    complete || error("Complete checkpoint did not restore as a complete chain.")
                    initial_results[i] = result
                elseif retained > 0
                    initial_stepsizes[i] = _adapted_stepsize(payload)
                    _mass_check(admission) &&
                        (initial_log_masses[i] = _payload_log_mass(payload, densities[i]))
                end
            end
        end
        out = _completion_batch(run_chain, n; min_completed=Int(min_completed), admission,
            initial_results, initial_stepsizes, initial_log_masses, progress=cprogress)
        out = (; out.results, completion=merge(out.completion, (; n_draws, attempt_directory)))
        _finish_completion_progress!(cprogress, out.completion)
        out
    end
    if outcome.completion.quorum_met && !isnothing(terminal_path)
        _atomic_serialize(terminal_path, (; schema_version=2, n_draws, outcome))
    end
    _write_completion_summary(attempt_directory, outcome.completion)
    # Only now is it known whether the retained chains are a runtime selection.
    _warn_runtime_selection(outcome.completion)
    outcome.completion.quorum_met || throw(CompletionQuorumError(outcome))
    outcome
end

completion_warmup_mcmc(rngs::AbstractVector, lpdf; kwargs...) =
    completion_warmup_mcmc(rngs, fill(lpdf, length(rngs)); kwargs...)
