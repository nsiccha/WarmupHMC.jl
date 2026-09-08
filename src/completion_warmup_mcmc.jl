# Completion is an orchestration policy over independent adaptive chains. It
# deliberately does not reuse the cooperative sampler's ESS/pool scheduler.
const _COMPLETION_SELECTION_WARNING = "Chains were selected by runtime completion. " *
    "Runtime can depend on the sampled states; retaining faster chains can bias inference. " *
    "R-hat, ESS and divergence checks on retained draws cannot rule out this selection bias."

"An unmet completion quorum. `outcome` contains the settled results and every chain's disposition."
struct CompletionQuorumError{T} <: Exception
    outcome::T
end
Base.showerror(io::IO, e::CompletionQuorumError) = print(io,
    "Completion quorum unmet: ", e.outcome.completion.n_completed, " of ",
    e.outcome.completion.min_completed, " required chains completed. Inspect `outcome.completion.chains`.")

mutable struct _CompletionState{C}
    lock::ReentrantLock
    clock::C
    start::Float64
    min_completed::Int
    grace_seconds::Float64
    quorum_at::Union{Nothing,Float64}
    cutoff_at::Union{Nothing,Float64}
    statuses::Vector{Symbol}
    results::Vector{Any}
    errors::Vector{Any}
    finished_at::Vector{Union{Nothing,Float64}}
end

_completion_elapsed(s::_CompletionState) = Float64(s.clock()) - s.start
_completion_expired(s::_CompletionState, now) =
    !isnothing(s.cutoff_at) && now >= s.cutoff_at

function _completion_stop(s::_CompletionState)
    lock(s.lock) do
        _completion_expired(s, _completion_elapsed(s))
    end
end

function _completion_worker!(run_chain, s::_CompletionState, i)
    started = lock(s.lock) do
        if s.statuses[i] === :completed
            false
        elseif _completion_expired(s, _completion_elapsed(s))
            s.statuses[i] = :not_started
            false
        else
            s.statuses[i] = :running
            true
        end
    end
    started || return
    try
        result, complete = run_chain(i, () -> _completion_stop(s))
        lock(s.lock) do
            now = _completion_elapsed(s)
            s.results[i] = result
            s.finished_at[i] = now
            s.statuses[i] = if !complete
                :stopped
            elseif _completion_expired(s, now)
                :completed_after_cutoff
            else
                :completed
            end
            if isnothing(s.quorum_at) && count(==(:completed), s.statuses) >= s.min_completed
                s.quorum_at = now
                s.cutoff_at = now + s.grace_seconds
            end
        end
    catch err
        # A failed chain cannot strand a worker or masquerade as a completion.
        # Preserve the exception AND backtrace, then join the other workers.
        failure = CapturedException(err, catch_backtrace())
        lock(s.lock) do
            s.errors[i] = failure
            s.finished_at[i] = _completion_elapsed(s)
            s.statuses[i] = :failed
        end
    end
    nothing
end

# The injected runner/clock are internal seams for controlled scheduling tests;
# the public method below always runs the real adaptive sampler and monotonic time.
function _completion_batch(run_chain, n; min_completed, grace_seconds,
        initial_results=Dict{Int,Any}(), clock=() -> time_ns() / 1e9)
    s = _CompletionState(ReentrantLock(), clock, Float64(clock()), min_completed,
        Float64(grace_seconds), nothing, nothing, fill(:pending, n),
        Any[nothing for _ in 1:n], Any[nothing for _ in 1:n],
        Union{Nothing,Float64}[nothing for _ in 1:n])
    # Resume admission is a deterministic pre-pass, not a race between tasks
    # restoring already-complete checkpoints. Admit ALL such original slots.
    for (i, result) in initial_results
        s.results[i] = result
        s.statuses[i] = :completed
        s.finished_at[i] = 0.0
    end
    if length(initial_results) >= min_completed
        s.quorum_at = 0.0
        s.cutoff_at = s.grace_seconds
    end
    @sync for i in 1:n
        Threads.@spawn _completion_worker!(run_chain, s, i)
    end
    # All workers are joined before any result or terminal file is published.
    completed = findall(==(:completed), s.statuses)
    failed = findall(==(:failed), s.statuses)
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
           error=s.errors[i])
    end
    n_samples = sum(r -> r.n_samples, results; init=0)
    n_divergent_samples = sum(r -> r.n_divergent_samples, results; init=0)
    quorum_met = length(completed) >= min_completed
    reason = !quorum_met ? :quorum_unmet : isempty(omitted) ? :all_completed :
        _completion_expired(s, _completion_elapsed(s)) ? :grace_expired : :all_settled
    completion = (; policy=:completion_quorum, min_completed, grace_seconds=Float64(grace_seconds),
        n_requested=n, n_started=length(started), n_completed=length(completed),
        n_failed=length(failed), n_omitted=length(omitted),
        requested_chain_indices=collect(1:n), started_chain_indices=started,
        completed_chain_indices=completed, failed_chain_indices=failed,
        omitted_chain_indices=omitted, chains, quorum_met, stop_reason=reason,
        quorum_at_seconds=s.quorum_at, cutoff_at_seconds=s.cutoff_at,
        elapsed_seconds=_completion_elapsed(s), n_samples, n_divergent_samples,
        selection_warning=_COMPLETION_SELECTION_WARNING)
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
        manifest.schema_version == 1 && manifest.sampler === :completion ||
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
        _atomic_serialize_manifest(manifest_path, (; schema_version=1, sampler=:completion, n_requested=n))
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
    _write_json(joinpath(dir, "run_summary.json"), fields)
end

"""
    completion_warmup_mcmc(rngs, lpdf_or_lpdfs; min_completed=length(rngs),
        grace_seconds=0.0, n_draws=1000, checkpoint_dir=nothing,
        resume=false, overwrite=false, kwargs...)

Opt-in completion quorum over independent [`adaptive_warmup_mcmc`](@ref)
chains. For example, pass 16 RNGs and `min_completed=12, grace_seconds=30`
to start a 30-second grace period when the twelfth full chain returns.
All chains completing before the cutoff are retained, in original index order.
If all workers settle earlier the call returns immediately. The default
samplers and their return types are unchanged.

Each chain runs the ordinary adaptive procedure with its own deep-copied
density, initialization and RNG slot. There is no cross-chain adaptation.
`init` accepts the adaptive multi-chain per-chain forms. Other `kwargs` are
adaptive single-chain keywords, including `callback(state, stage)`: returning
`true` there stops only that chain and does not count an incomplete chain.
Callbacks may run concurrently and must be observational and thread-safe.

Grace requests a cooperative stop at the next checkpoint boundary. An active
initialization or adaptive window must finish; this is **not a hard wall-time
limit**. No tasks are interrupted or detached, and every worker is joined
before return. A full chain finishing after the cutoff is also omitted.
Julia threads are required for simultaneous CPU work; runtime selection and
completion order are not reproducible from RNG seeds alone.

Returns `(; results, completion)`. Only admitted full chains appear in
`results`, each carrying the adaptive result fields plus original `chain_index`
and actual `n_samples`. `completion` records policy, requested/started/completed/
failed/omitted identities, all chain dispositions and captured errors, quorum
and cutoff times (seconds since worker scheduling begins), and retained
draw/divergence counts. A failed chain's `n_samples` is `nothing` because it
has no finalized result; its `n_retained_samples` is zero. Failed chains are
also omitted. If the quorum is unmet, throws
`WarmupHMC.CompletionQuorumError` with this same `outcome`; partial results are
never silently reported as a successful fit.

**Inferential limitation:** selecting faster chains can bias inference when
runtime depends on sampled states or modes. R-hat, bulk/tail ESS and divergence
checks describe only the returned draws and cannot rule out that bias. Report
the policy and all omissions; do not describe the requested count as completed.
Use all-chain sampling when this selection is unacceptable.

Checkpoint files remain ordinary adaptive source-frame state under original
`chain_<i>` directories. `resume=true` requires the same ordered RNG/density
slots; existing checkpoints restore their saved RNG, while a slot that never
checkpointed initializes from the supplied RNG. Re-pass original sampler
settings. For an interrupted run, already-complete checkpoints are admitted
deterministically before workers launch, then a fresh grace period starts if
they meet quorum. A successful terminal run reopens its **recorded outcome**:
it never samples or admits omitted chains, and changing `n_draws`,
`min_completed` or `grace_seconds` is rejected. Extend individual adaptive
checkpoints separately when a different production target is required.
`completion_manifest.jls` binds
the original slot count; each invocation writes its own `run_manifest.json`
and, after all workers settle, `run_summary.json` under
`completion_attempts/attempt_<n>/`. The returned `completion.attempt_directory`
identifies that invocation. `completion_terminal.jls` stores the successful
outcome, including returned draws, atomically before the summary is published.
These run records do not change chain payloads.
"""
function completion_warmup_mcmc(rngs::AbstractVector, lpdfs::AbstractVector;
        min_completed=length(rngs), grace_seconds=0.0, n_draws=1000,
        init=missing, checkpoint_dir=nothing, resume=false, overwrite=false,
        callback=nothing, kwargs...)
    n = length(rngs)
    Base.require_one_based_indexing(rngs, lpdfs)
    n > 0 || throw(ArgumentError("At least one RNG is required."))
    length(lpdfs) == n || throw(DimensionMismatch("One density per RNG is required."))
    min_completed isa Integer && !(min_completed isa Bool) && 1 <= min_completed <= n ||
        throw(ArgumentError("`min_completed` must be an integer in 1:$n."))
    grace_seconds isa Real && isfinite(grace_seconds) && grace_seconds >= 0 ||
        throw(ArgumentError("`grace_seconds` must be finite and nonnegative."))
    isfinite(Float64(grace_seconds)) || throw(ArgumentError("`grace_seconds` must fit in Float64 seconds."))
    n_draws isa Integer && !(n_draws isa Bool) && 0 < n_draws < typemax(Int) ||
        throw(ArgumentError("`n_draws` must be a finite positive integer."))
    _check_kwargs(:completion_warmup_mcmc, kwargs)
    inits = ensurevector(init, n)
    resume && overwrite && throw(ArgumentError("`resume` and `overwrite` are mutually exclusive."))
    terminal_path = isnothing(checkpoint_dir) ? nothing : joinpath(checkpoint_dir, "completion_terminal.jls")
    if resume && !isnothing(terminal_path) && isfile(terminal_path)
        saved = deserialize(terminal_path)
        saved.schema_version == 1 || throw(ArgumentError("Unsupported terminal completion schema."))
        c = saved.outcome.completion
        (c.n_requested == n && saved.n_draws == n_draws &&
            c.min_completed == min_completed && c.grace_seconds == grace_seconds) ||
            throw(ArgumentError("A finished completion run has an immutable target, policy and chain selection. " *
                "Reopen with its original configuration or use a new directory."))
        # Recover a crash after the atomic terminal write but before its JSON
        # publication, without reopening the selection decision.
        isfile(joinpath(c.attempt_directory, "run_summary.json")) ||
            _write_completion_summary(c.attempt_directory, c)
        @warn _COMPLETION_SELECTION_WARNING
        return saved.outcome
    end
    # Copies are made serially, before any worker can mutate its density.
    densities = map(deepcopy, lpdfs)
    attempt_directory = _completion_directory(checkpoint_dir, n, resume, overwrite)
    if !isnothing(attempt_directory)
        _write_json(joinpath(attempt_directory, "run_manifest.json"), Pair{String,Any}[
            "schema_version" => 1, "sampler" => "completion", "n_requested" => n,
            "min_completed" => min_completed, "grace_seconds" => grace_seconds,
            "n_draws" => n_draws, "resume" => resume,
            "selection_warning" => _COMPLETION_SELECTION_WARNING])
    end
    @warn _COMPLETION_SELECTION_WARNING
    run_chain = function (i, should_stop)
        dir = _chain_dir(checkpoint_dir, i)
        resume_chain = resume && !isempty(_checkpoint_files(dir))
        boundary = (state, stage) -> begin
            user_stop = _fire_callback(callback, state, stage)
            should_stop() || user_stop
        end
        result = adaptive_warmup_mcmc(rngs[i], densities[i]; n_draws, init=deepcopy(inits[i]),
            checkpoint_dir=dir, resume=resume_chain, callback=boundary, kwargs...)
        result, size(result.posterior_position, 2) >= n_draws
    end
    initial_results = Dict{Int,Any}()
    if resume
        for i in 1:n
            latest = joinpath(_chain_dir(checkpoint_dir, i), "cp_latest.jls")
            isfile(latest) || continue
            payload = deserialize(latest)
            if size(payload.posterior_position, 2) >= n_draws
                result, complete = run_chain(i, () -> false)
                complete || error("Complete checkpoint did not restore as a complete chain.")
                initial_results[i] = result
            end
        end
    end
    outcome = _completion_batch(run_chain, n; min_completed=Int(min_completed), grace_seconds, initial_results)
    outcome = (; outcome.results, completion=merge(outcome.completion, (; n_draws, attempt_directory)))
    if outcome.completion.quorum_met && !isnothing(terminal_path)
        _atomic_serialize(terminal_path, (; schema_version=1, n_draws, outcome))
    end
    _write_completion_summary(attempt_directory, outcome.completion)
    outcome.completion.quorum_met || throw(CompletionQuorumError(outcome))
    outcome
end

completion_warmup_mcmc(rngs::AbstractVector, lpdf; kwargs...) =
    completion_warmup_mcmc(rngs, fill(lpdf, length(rngs)); kwargs...)
