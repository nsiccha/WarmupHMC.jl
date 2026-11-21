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
    _stuck_note(e.outcome.completion), _failed_note(e.outcome.completion),
    ". Inspect `outcome.completion.chains`.")
# A run recorded before the admission checks has no stuck chains to name. A chain
# left out while it was still warming up was judged before it reached `n_draws`,
# so it carries no `divergent_fraction`.
_stuck_note(c) = begin
    stuck = get(c, :stuck_chain_indices, Int[])
    isempty(stuck) && return ""
    warming = Int[ch.chain_index for ch in get(c, :chains, ())
                  if ch.status === :stuck && isnothing(get(ch, :divergent_fraction, 0.0))]
    finished = Int[i for i in stuck if i ∉ warming]
    (isempty(finished) ? "" : "; left out as stuck after reaching `n_draws`: $finished") *
    (isempty(warming) ? "" :
        "; left out during warm-up, their step size collapsed below their siblings': $warming")
end
# A failed chain's exception is otherwise visible only in `outcome.completion.chains`,
# so the quorum error carries the first line of each (snag centred-latent-l-16f7838d:
# a chain whose target broke failed with a message that named the cause).
_failed_note(c) = join(("; chain $(ch.chain_index) failed: $(_first_error_line(ch.error))"
                        for ch in get(c, :chains, ())
                        if ch.status === :failed && !isnothing(get(ch, :error, nothing))), "")
_first_error_line(e::CapturedException) = _first_error_line(e.ex)
_first_error_line(e) = first(split(sprint(showerror, e), '\n'))

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
# A chain still warming up is left out once its step size at each of its last two
# window boundaries is below the median of its siblings that keep draws by
# `stuck_stepsize_ratio` times this further factor (snag collapsed-chains-10d2df94).
# Measured on 393 window boundaries of healthy chains (11 targets: Gaussians to
# d = 200, a correlated Gaussian, centred funnels at d = 10 and 50, centred eight
# schools with 8 and 16 chains, a logistic regression, Student-t, a banana and a
# mixture of unequal widths; 8 or 16 chains each, the median of the siblings'
# latest step sizes kept with draws as reference): the larger of a chain's last
# two window-boundary step sizes was at most 2.6 times below that median, a
# single boundary at most 6.1 times. Chains that collapsed sat 1e3 to 1e16 times
# below, and one that started in a noisy region dipped 180 times below at its
# first boundary and recovered at its second, which is why two boundaries count.
const _WARMUP_STEPSIZE_MARGIN = 10.0
const _LOG_MASS_GAP_STANDARD_ERRORS = 4.0
const _COMPLETION_RESULT_SCHEMA = 5  # common final reference and joint mass uncertainty
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

# Correlation time of a contribution to the mass score. A constant series
# places no constraint on the block length. A failed estimate cannot validate
# an exclusion.
function _mass_correlation_time(x)
    allequal(x) && return 1.0
    ess = MCMCDiagnosticTools.ess(reshape(x, :, 1, 1); kind=:basic)[1]
    isfinite(ess) && ess > 0 ? max(1.0, length(x) / ess) : Inf
end

# Delete contiguous blocks from the *whole* nonlinear score. Center around
# the full mean when subtracting moments. Only one block's O(d) sums are live.
function _mass_jackknife(draw, lp, mu, m2)
    n = length(lp)
    mean_lp = mean(lp)
    # First-order fluctuation of the log marginal volume at each draw.
    volume = similar(lp)
    for j in eachindex(lp)
        _, x = draw(j)
        volume[j] = (n - 1) / 2 * sum(k -> abs2(x[k] - mu[k]) / m2[k], eachindex(mu))
    end
    tau = max(_mass_correlation_time(lp), _mass_correlation_time(volume),
        _mass_correlation_time(lp .- mean_lp .+ volume))
    # Size blocks for both parts even when their fluctuations cancel. At least
    # eight blocks are required here to estimate the sampling error.
    width = max(sqrt(n), 2tau)
    blocks = floor(Int, n / width)
    blocks >= 8 || return nothing
    offsets = [fld(b * n, blocks) for b in 0:blocks]
    pseudovalues = zeros(blocks)
    sums, squares = similar(mu), similar(m2)
    for b in 1:blocks
        rows = (offsets[b] + 1):offsets[b + 1]
        count = length(rows)
        remaining = n - count
        fill!(sums, 0)
        fill!(squares, 0)
        for j in rows
            _, x = draw(j)
            for k in eachindex(mu)
                z = x[k] - mu[k]
                sums[k] += z
                squares[k] += z * z
            end
        end
        delta = -sum(j -> lp[j] - mean_lp, rows) / remaining
        for k in eachindex(mu)
            removed = (squares[k] + sums[k]^2 / remaining) / m2[k]
            removed < 1 || return nothing  # zero variance after deletion
            delta += (log1p(-removed) + log1p(count / (remaining - 1))) / 2
        end
        # The full score is a common additive constant and cancels in the SE.
        pseudovalues[b] = -remaining / count * delta
    end
    weights = diff(offsets) ./ n
    center = sum(weights .* pseudovalues)
    se = sqrt(sum(b -> weights[b] * abs2(pseudovalues[b] - center), 1:blocks) / (blocks - 1))
    isfinite(se) ? se : nothing
end

# Mean model-frame log density plus the log marginal volume. Use a joint
# block error because coordinates and density can fluctuate together.
function _log_mass_estimate(ir, positions, log_density)
    d, n = size(positions)
    length(log_density) == n || throw(DimensionMismatch(
        "one log density per draw is required; got $(length(log_density)) for $n draws"))
    n > _MIN_LOG_MASS_DRAWS || return nothing
    mapped = !isnothing(ir) && has_reparametrization(ir)
    draw(j) = begin
        x = view(positions, :, j)
        mapped ? ir(x) : (0.0, x)
    end
    lp = Vector{Float64}(undef, n)
    mu, m2 = zeros(d), zeros(d)
    for j in 1:n
        ljac, y = draw(j)
        lp[j] = log_density[j] - ljac
        for k in 1:d
            delta = y[k] - mu[k]
            mu[k] += delta / j
            m2[k] += delta * (y[k] - mu[k])
        end
    end
    all(>(0), m2) || return nothing
    log_mass = mean(lp) + sum(v -> log(v / (n - 1)) / 2, m2)
    isfinite(log_mass) || return nothing
    (; log_mass, se=_mass_jackknife(draw, lp, mu, m2))
end

# Finished results have already been transformed into the common model frame.
_result_log_mass(r) = hasproperty(r, :posterior_log_density) ?
    _log_mass_estimate(nothing, r.posterior_position, r.posterior_log_density) : nothing

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
    changed::Threads.Condition
    clock::C
    start::Float64
    min_completed::Int
    admission::_CompletionAdmission
    provisional_at::Union{Nothing,Float64}
    quorum_at::Union{Nothing,Float64}
    at_boundary::Vector{Bool}
    statuses::Vector{Symbol}
    results::Vector{Any}
    errors::Vector{Any}
    finished_at::Vector{Union{Nothing,Float64}}
    # Each slot's adapted step size (`_adapted_stepsize`) at its latest checkpoint
    # boundary, or `nothing` while it
    # keeps no draws there (still initializing, or its window just restarted).
    sampling_stepsizes::Vector{Union{Nothing,Float64}}
    # Each slot's adapted step size at its latest and at its previous warm-up
    # WINDOW boundary, whether or not it keeps draws there: what the warm-up check
    # judges (`_judge_warming!`). A chain whose step size collapsed can keep no
    # draws at any boundary, and then never publishes a sampling step size.
    warming_stepsizes::Vector{Union{Nothing,Float64}}
    previous_warming_stepsizes::Vector{Union{Nothing,Float64}}
    # Each judged slot's admission record; `_UNJUDGED` until it reaches `n_draws`.
    admissions::Vector{Any}
end

_CompletionState(n; clock, min_completed, admission=_ADMIT_EVERY_FULL_CHAIN) =
    _CompletionState(Threads.Condition(), clock, Float64(clock()), min_completed, admission,
        nothing, nothing, fill(false, n), fill(:pending, n), Any[nothing for _ in 1:n], Any[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n],
        Any[_UNJUDGED for _ in 1:n])

_completion_elapsed(s::_CompletionState) = Float64(s.clock()) - s.start
# A provisional quorum becomes final once every running worker has either
# delivered its round's result or reached a boundary with an incomplete chain.
# Pending slots can wait without acquiring a worker or initializing a chain.
function _confirm_quorum!(s)
    if !isnothing(s.provisional_at) && (!_mass_check(s.admission) ||
            all(i -> s.statuses[i] !== :running || s.at_boundary[i], eachindex(s.statuses)))
        s.quorum_at = s.provisional_at
    end
end

# Caller holds the condition lock. Waiting yields the Julia task, including on
# a single thread. The adaptive sampler and RNG remain on its call stack.
function _stop_at_boundary!(s, i)
    if !isnothing(s.provisional_at) && isnothing(s.quorum_at)
        s.at_boundary[i] = true
        try
            _confirm_quorum!(s)
            notify(s.changed)
            while !isnothing(s.provisional_at) && isnothing(s.quorum_at) && s.statuses[i] !== :stuck
                wait(s.changed)
            end
        finally
            s.at_boundary[i] = false
        end
    end
    !isnothing(s.quorum_at) || s.statuses[i] === :stuck
end

struct _CompletionControl{S,P}
    state::S
    slot::Int
    progress::P
end
(c::_CompletionControl)() = lock(() -> _stop_at_boundary!(c.state, c.slot), c.state.changed)

# Whether the chain was left out while it ran (`_judge_warming!`). Polled after
# every transition, so a chain left out in the middle of a long window stops
# there instead of finishing it. The resume pre-pass's restore control is a plain
# function, and is never left out.
_left_out(c::_CompletionControl) = lock(() -> c.state.statuses[c.slot] === :stuck, c.state.changed)
_left_out(_) = false

# `window` marks a warm-up window boundary (not initialization): there the
# chain's step size is published for the warm-up check whether or not it keeps
# draws.
function _report_boundary!(c::_CompletionControl, stepsize, n_samples;
        window=false)
    s = c.state
    changed = lock(s.changed) do
        before = copy(s.statuses)
        s.sampling_stepsizes[c.slot] = n_samples > 0 ? Float64(stepsize) : nothing
        if window
            s.previous_warming_stepsizes[c.slot] = s.warming_stepsizes[c.slot]
            s.warming_stepsizes[c.slot] = Float64(stepsize)
        end
        _settle_admissions!(s, _completion_elapsed(s))
        s.statuses != before
    end
    changed && _update_completion_progress!(c.progress, s)
    nothing
end

# Register the full result. Divergence admission needs no sibling. Once the
# cohort has seen the new step size, _settle_admissions! handles the other checks.
function _admit!(s::_CompletionState, i, own)
    r = s.results[i]
    s.admissions[i] = (; stuck_reason=nothing, stepsize=_adapted_stepsize(r),
        reference_stepsize=nothing,
        divergent_fraction=r.n_divergent_samples / size(r.posterior_position, 2),
        log_mass=isnothing(own) ? nothing : own.log_mass,
        log_mass_se=isnothing(own) ? nothing : own.se,
        reference_log_mass=nothing, reference_log_mass_se=nothing)
    if s.admissions[i].divergent_fraction > s.admission.max_admitted_divergent_fraction
        _leave_out!(s, i, :divergence)
    else
        s.statuses[i] = :awaiting_reference
    end
    nothing
end

function _judge_stepsize!(s, i)
    if isinf(s.admission.stuck_stepsize_ratio)
        s.statuses[i] = :completed
        return
    end
    reference = _sibling_stepsize(s, i)
    if isnothing(reference)
        any(st -> st in (:pending, :running), s.statuses) && return
    else
        a = s.admissions[i]
        s.admissions[i] = merge(a, (; reference_stepsize=reference))
        a.stepsize < reference / s.admission.stuck_stepsize_ratio &&
            return _leave_out!(s, i, :stepsize)
    end
    s.statuses[i] = :completed
end

# Revisit both previously passing and previously low-mass finishers. A new
# best score with a larger error can also *restore* an earlier exclusion.
function _judge_masses!(s)
    scored = [i for i in eachindex(s.statuses) if
        (s.statuses[i] === :completed || s.admissions[i].stuck_reason === :low_mass) &&
        !isnothing(s.admissions[i].log_mass)]
    length(scored) >= 2 || return
    # argmax breaks ties by the first slot, independently of arrival order.
    best = s.admissions[argmax(i -> s.admissions[i].log_mass, scored)]
    for i in scored
        a = s.admissions[i]
        a = merge(a, (; reference_log_mass=best.log_mass, reference_log_mass_se=best.log_mass_se))
        low = _low_mass(a, s.admission)
        s.admissions[i] = merge(a, (; stuck_reason=low ? :low_mass : nothing))
        s.statuses[i] = low ? :stuck : :completed
    end
    nothing
end

# Both bounds must be exceeded: the gap in nats, and the gap in combined
# standard errors, which keeps a noisy estimate such as a funnel's.
_low_mass(a, admission) = begin
    gap = a.reference_log_mass - a.log_mass
    !isnothing(a.log_mass_se) && !isnothing(a.reference_log_mass_se) &&
        gap > admission.max_admitted_log_mass_gap &&
        gap > _LOG_MASS_GAP_STANDARD_ERRORS * hypot(a.log_mass_se, a.reference_log_mass_se)
end

function _leave_out!(s::_CompletionState, i, reason)
    s.admissions[i] = merge(s.admissions[i], (; stuck_reason=reason))
    s.statuses[i] = :stuck
    nothing
end

# Mass verdicts are provisional. Only a failure or a step-size/divergence
# exclusion disqualifies a sibling as a step-size reference.
function _sibling_stepsize(s::_CompletionState, i)
    steps = Float64[s.sampling_stepsizes[k] for k in eachindex(s.statuses)
        if k != i && !isnothing(s.sampling_stepsizes[k]) && (s.statuses[k] ∉ (:stuck, :failed) ||
            s.admissions[k].stuck_reason === :low_mass)]
    isempty(steps) ? nothing : median(steps)
end

# Must hold `s.changed`. A chain still warming up (running, or a resumed slot not
# yet restarted) whose step size at each of its last two window boundaries is
# below its siblings' median by `stuck_stepsize_ratio * _WARMUP_STEPSIZE_MARGIN`
# is left out at once, before it reaches `n_draws`. Its step size and metric have collapsed:
# each of its transitions then runs to the tree-depth ceiling, its windows double,
# and it would hold its thread for hours before the check at `n_draws` could
# judge it (snag collapsed-chains-10d2df94: step sizes of 1e-11 to 1e-7, 920 to
# 1000 gradients per draw, 11 hours to go). WarmupHMC cannot reliably move such
# a chain back (user principle `caller-recovers-lost-chain`), so it is stopped
# and left out, as the user asked the quorum sampler to do with a chain stuck in
# warm-up (`quorum-sampler-everywhere`). The reference is the step-size check's:
# siblings keeping draws that still qualify on step size and divergence.
function _judge_warming!(s::_CompletionState, i)
    ratio = s.admission.stuck_stepsize_ratio * _WARMUP_STEPSIZE_MARGIN
    isfinite(ratio) || return nothing
    stepsize, previous = s.warming_stepsizes[i], s.previous_warming_stepsizes[i]
    (isnothing(stepsize) || isnothing(previous)) && return nothing
    reference = _sibling_stepsize(s, i)
    (isnothing(reference) || max(stepsize, previous) >= reference / ratio) && return nothing
    s.admissions[i] = merge(_UNJUDGED, (; stuck_reason=:stepsize, stepsize,
        reference_stepsize=reference))
    s.statuses[i] = :stuck
    # No longer a reference for its siblings.
    s.sampling_stepsizes[i] = nothing
    nothing
end

# Must hold `s.changed`. Judge every warming slot, then re-judge every awaiting slot
# after any change a judgement depends on, then record when the quorum is first
# met. Warming slots come first: a collapsed chain that still keeps draws would
# otherwise pull down the reference an awaiting slot is compared with.
function _settle_admissions!(s::_CompletionState, now)
    for i in eachindex(s.statuses)
        s.statuses[i] in (:pending, :running) && _judge_warming!(s, i)
    end
    new_finisher = false
    for i in eachindex(s.statuses)
        s.statuses[i] === :awaiting_reference || continue
        _judge_stepsize!(s, i)
        new_finisher |= s.statuses[i] === :completed
    end
    new_finisher && _mass_check(s.admission) && _judge_masses!(s)
    if count(==(:completed), s.statuses) >= s.min_completed
        s.provisional_at = something(s.provisional_at, now)
        _confirm_quorum!(s)
    else
        s.provisional_at = nothing
    end
    notify(s.changed)
    nothing
end

function _completion_worker!(run_chain, s::_CompletionState, i; progress=nothing)
    started = lock(s.changed) do
        if s.statuses[i] in (:completed, :stuck, :awaiting_reference)
            # Already judged by the resume pre-pass.
            false
        elseif _stop_at_boundary!(s, i) || _interrupted(progress)
            # Quorum met, or a stop requested through the progress tree before
            # this slot started: never pay a fresh chain's initialization.
            s.statuses[i] === :stuck || (s.statuses[i] = :not_started)
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
            own = complete ? _result_log_mass(result) : nothing
            lock(s.changed) do
                now = _completion_elapsed(s)
                s.results[i] = result
                s.finished_at[i] = now
                if s.statuses[i] !== :stuck
                    s.sampling_stepsizes[i] = size(result.posterior_position, 2) > 0 ?
                        _adapted_stepsize(result) : nothing
                    # Quorum requests a stop at the next checkpoint; it never closes
                    # admission. A full result from the final round is still judged.
                    complete ? _admit!(s, i, own) : (s.statuses[i] = :stopped)
                end
                # A chain left out while it warmed up keeps that verdict.
                _settle_admissions!(s, now)
            end
        catch err
            # A failed chain cannot strand a worker or masquerade as a completion.
            # Preserve the exception AND backtrace, then join the other workers.
            failure = CapturedException(err, catch_backtrace())
            lock(s.changed) do
                now = _completion_elapsed(s)
                s.errors[i] = failure
                s.finished_at[i] = now
                # A chain already left out keeps that verdict; its error is kept too.
                s.statuses[i] === :stuck || (s.statuses[i] = :failed)
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
    snap = lock(s.changed) do
        (; n_completed=count(==(:completed), s.statuses),
           stop_requested=!isnothing(s.provisional_at) || !isnothing(s.quorum_at),
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
        initial_warming_stepsizes=Dict{Int,NTuple{2,Float64}}(),
        clock=() -> time_ns() / 1e9, progress=nothing)
    s = _CompletionState(n; clock, min_completed, admission)
    # A partial checkpoint past two window boundaries is judged by the warm-up
    # check before its slot resumes, so a chain left out before the run stopped is
    # not restarted. Each entry is (previous, latest).
    for (i, (previous, latest)) in initial_warming_stepsizes
        s.previous_warming_stepsizes[i] = previous
        s.warming_stepsizes[i] = latest
    end
    # Publish all restored step sizes and full results before judging the
    # cohort. Dictionary iteration order cannot set the mass reference.
    for (i, stepsize) in initial_stepsizes
        s.sampling_stepsizes[i] = stepsize
    end
    for (i, result) in initial_results
        s.results[i] = result
        s.finished_at[i] = 0.0
        s.sampling_stepsizes[i] = _adapted_stepsize(result)
        _admit!(s, i, _result_log_mass(result))
    end
    lock(() -> _settle_admissions!(s, 0.0), s.changed)
    _update_completion_progress!(progress, s)
    @sync for i in 1:n
        Threads.@spawn _completion_worker!(run_chain, s, i; progress)
    end
    # All workers are joined before any result or terminal file is published.
    # With no slot left running, every awaiting slot is judged here at the latest.
    lock(() -> _settle_admissions!(s, _completion_elapsed(s)), s.changed)
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
    completion = (; schema_version=_COMPLETION_RESULT_SCHEMA, policy=:completion_quorum,
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

Run independent [`adaptive_warmup_mcmc`](@ref) chains until enough full chains
pass admission. For example, with 16 RNGs and `min_completed=12`, reaching
12 passing chains asks each running worker to finish its current round.
Additional full results from that round participate in the same mass comparison.
If the passing count drops below 12, unfinished workers continue from their
round boundaries. Results are returned in original slot order after workers settle.

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
- its estimated log posterior mass passes a comparison with the highest score
  among full chains that pass the first two checks. A chain is left out if its
  gap exceeds both `max_admitted_log_mass_gap` nats and four combined standard
  errors. The score adds the mean model-frame log density to the sum of the log
  marginal standard deviations. Its error comes from deleting contiguous blocks
  of draws and recomputing both terms together. A comparison requires a finite
  score and an estimable error for both chains.

A chain failing a check is marked `:stuck` and omitted. A chain can wait for
a sampling sibling's step size. The mass check revisits earlier finishers
when the reference changes. For a fixed set of otherwise eligible full
results its verdicts are independent of finish order. Early stopping can
change which slots finish. The score ignores differences in correlation
structure between modes, and a run entirely in one minor mode has no other
mode to compare with.

**A chain whose step size collapses while it warms up is left out then, not at
`n_draws`.** At every warm-up window boundary, its own or a sibling's, a running
chain whose step size at each of its last two window boundaries is below the
median of its siblings that keep draws by more than `stuck_stepsize_ratio` times
a further factor of 10 (100 at the default) is `:stuck` with
`stuck_reason = :stepsize`, and stops at its next transition. Such a chain's
step size and metric have collapsed: each transition runs to the tree-depth
ceiling, its windows double, and it would otherwise hold its thread for hours
before reaching `n_draws`.
WarmupHMC does not move it back; rerun it from another start if it is needed.
Its `completion.chains` entry records its latest window-boundary `stepsize`, the
`reference_stepsize` it was judged by, and no `divergent_fraction`. On resume, a
partial checkpoint written at a window boundary, with the previous window's
checkpoint beside it, is judged the same way before its slot restarts.
`stuck_stepsize_ratio=Inf`, `max_admitted_divergent_fraction=1` and
`max_admitted_log_mass_gap=Inf` switch the checks off (the first switches off
the warm-up check too).

Each chain runs the ordinary adaptive procedure with its own deep-copied
density, initialization and RNG slot. There is no cross-chain adaptation.
`init` accepts the adaptive multi-chain per-chain forms. Other `kwargs` are
adaptive single-chain keywords, including `callback(state, stage)`: returning
`true` there stops only that chain and does not count an incomplete chain, as
does a `stop_requested` predicate, combined with the warm-up check's own.
`Treebars.request_interrupt!` behaves the same way through the progress tree: on
one chain's node it stops that chain, on the node passed as `progress` (or an
ancestor) it stops every running chain and starts no further slot.
Callbacks may run concurrently and must be observational and thread-safe.
`scale_centering = true` runs each chain through both passes of
[`adaptive_warmup_mcmc`](@ref)'s scale-dependent centering: pass 1 selects the chain's centering, pass 2
samples it from a fresh start, and its result carries `selected_centering` and
`first_pass`. The quorum stops a chain at a boundary of either pass; a chain
stopped in pass 1 runs no pass 2 and is `:stopped`. The admission checks judge
pass 2 only: pass 1's step sizes and draws are not published to siblings. The
two passes' limits apply (no `checkpoint_dir`, `resume`, `overwrite`, nor a
`PathfinderResult` `init`), and are refused before any chain starts.
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
`log_mass_se`, and the common `reference_log_mass` and
`reference_log_mass_se`. A missing score or error is `nothing`. A reference
is recorded for scored eligible finishers when at least two have scores
and the check is enabled. The record includes the reference chain itself.
Completion result schema 5 identifies this shared reference and joint error.
A stuck chain records `stuck_reason` (`:divergence`, `:stepsize` or `:low_mass`). A chain left out while it warmed up records only
`stuck_reason = :stepsize`, `stepsize` and `reference_stepsize`. A failed chain's `n_samples` is `nothing` because it
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
together before workers launch. Partial checkpoints can supply a step size.
Only full results can supply the mass reference. A quorum from the restored
results leaves the remaining slots unstarted. A successful terminal run reopens its **recorded outcome**, and
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
        scale_centering::Bool=false, kwargs...) = _with_blas_threads(blas_threads) do
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
    # The caller's own stop predicate is combined with the warm-up check's.
    caller_stop_requested = get(kwargs, :stop_requested, nothing)
    isnothing(caller_stop_requested) || applicable(caller_stop_requested) ||
        throw(ArgumentError("stop_requested must be a callable with no arguments"))
    chain_kwargs = Base.structdiff(values(kwargs), NamedTuple{(:stop_requested,)})
    inits = ensurevector(init, n)
    # Refused here, before any chain starts, rather than as n failed chains.
    if scale_centering
        checked = (; (k => kwargs[k] for k in (:nonlinear_adapt, :nonlinear_evidence,
                                                :checkpoint_interval) if haskey(kwargs, k))...)
        for i in 1:n
            _check_scale_centering(; init=inits[i], checkpoint_dir, resume, overwrite, checked...)
            _scale_centering_problem(lpdfs[i])
        end
    end
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
            "schema_version" => _COMPLETION_RESULT_SCHEMA, "sampler" => "completion", "n_requested" => n,
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
                # A sampling step size can settle a sibling waiting at n_draws.
                # `scale_centering`'s pass 1 publishes nothing: its draws are
                # discarded and its step sizes belong to another frame, so only
                # pass 2 is judged. Its boundaries still answer the quorum stop.
                _is_first_pass_stage(stage) || _report_boundary!(control, _adapted_stepsize(state),
                    size(state.recording_lpdf.posterior_position, 2);
                    window=stage === :window)
                # A full chain must deliver its result before the provisional
                # stop can be confirmed by workers with incomplete rounds.
                full = size(state.recording_lpdf.posterior_position, 2) >= n_draws
                user_stop || (!full && control())
            end
            # Polled after every transition: a chain left out while it warms up
            # stops there, not at the end of its window, which for a collapsed
            # chain is hours away. The quorum's own stop still waits for the
            # boundary (it finishes the current round).
            stop_requested = () -> _left_out(control) ||
                _external_stop_requested(caller_stop_requested)
            result = adaptive_warmup_mcmc(rngs[i], densities[i]; n_draws, init=deepcopy(inits[i]),
                checkpoint_dir=dir, resume=resume_chain, callback=boundary,
                progress=cprogress, description=description * ".$i", monitor_ess, blas_threads,
                scale_centering, stop_requested, chain_kwargs...)
            result, size(result.posterior_position, 2) >= n_draws
        end
        initial_results = Dict{Int,Any}()
        initial_stepsizes = Dict{Int,Float64}()
        initial_warming_stepsizes = Dict{Int,NTuple{2,Float64}}()
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
                else
                    if retained > 0
                        initial_stepsizes[i] = _adapted_stepsize(payload)
                    end
                    # Written at a window boundary, as the live check reads it (not
                    # at initialization, nor inside a window), with the previous
                    # window's boundary checkpoint beside it.
                    k = get(payload, :outer_counter, 0)
                    previous = joinpath(_chain_dir(checkpoint_dir, i), "cp_window_$(k - 1).jls")
                    k >= 2 && !get(payload, :window_in_progress, false) && isfile(previous) &&
                        (initial_warming_stepsizes[i] =
                            (_adapted_stepsize(deserialize(previous)), _adapted_stepsize(payload)))
                end
            end
        end
        out = _completion_batch(run_chain, n; min_completed=Int(min_completed), admission,
            initial_results, initial_stepsizes, initial_warming_stepsizes,
            progress=cprogress)
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
