# Reconstruct only the transform. Native target/AD handles need not be copyable,
# and candidate selection is never invoked while re-scoring retained points.
function _checkpoint_scoring_target(p::ReparametrizedProblem, ir)
    private_ir = isnothing(ir) ? snapshot_reparametrization(p.reparametrizer) : ir
    ReparametrizedProblem(private_ir, p.problem, p.ad_backend)
end
_checkpoint_scoring_target(p::Union{NamedPosterior,CountingPosterior,RecordingPosterior2}, ir) =
    _checkpoint_scoring_target(parent(p), ir)
function _checkpoint_scoring_target(p, ir)
    isnothing(ir) || throw(ArgumentError(
        "checkpoint re-scoring requires a ReparametrizedProblem to restore nonempty reparametrization controls"))
    p
end

mutable struct CheckpointRescoreAudit
    n_evaluations::Int
    changed_values::Int
    changed_gradient_components::Int
    max_abs::Float64
    max_scaled::Float64
    location::NamedTuple{(:field, :column, :coordinate),Tuple{Symbol,Int,Int}}
end
CheckpointRescoreAudit() = CheckpointRescoreAudit(0, 0, 0, 0.0, 0.0,
    (; field=:position_and_gradient, column=0, coordinate=0))

function _rescore_finite(x, location)
    isfinite(x) || throw(ArgumentError("checkpoint re-scoring found nonfinite data at $location"))
    nothing
end

function _rescore_compare!(audit, old, new, field, column, coordinate)
    location = (; field, column, coordinate)
    _rescore_finite(old, location)
    _rescore_finite(new, location)
    deviation = abs(new - old)
    scaled = deviation / max(abs(old), 1)
    audit.max_abs = max(audit.max_abs, deviation)
    if scaled > audit.max_scaled
        audit.max_scaled = scaled
        audit.location = location
    end
    if old != new
        if iszero(coordinate)
            audit.changed_values += 1
        else
            audit.changed_gradient_components += 1
        end
    end
end

function _rescore_evaluate!(audit, target, position, gradient, field, column, dimension)
    length(position) == length(gradient) == dimension || throw(DimensionMismatch(
        "checkpoint re-scoring: $field column $column must have $dimension coordinates"))
    for (i, x) in enumerate(position)
        _rescore_finite(x, (; field, column, coordinate=i))
    end
    # A callback may use scratch storage: give it an independent input and copy
    # its gradient before another evaluation can overwrite that storage.
    value, fresh_gradient = LogDensityProblems.logdensity_and_gradient(target, copy(position))
    _rescore_finite(value, (; field, column, coordinate=0))
    fresh_gradient isa AbstractVector && length(fresh_gradient) == dimension ||
        throw(DimensionMismatch("checkpoint re-scoring: target gradient at $field column $column must have $dimension coordinates"))
    for i in eachindex(gradient, fresh_gradient)
        _rescore_compare!(audit, gradient[i], fresh_gradient[i], field, column, i)
    end
    copyto!(gradient, fresh_gradient)
    audit.n_evaluations += 1
    value
end

function _rescore_matrix!(audit, target, positions, gradients, field, dimension)
    size(positions) == size(gradients) && size(positions, 1) == dimension ||
        throw(DimensionMismatch("checkpoint re-scoring: $field and its positions must have matching dimension × count shapes"))
    for column in axes(positions, 2)
        _rescore_evaluate!(audit, target, view(positions, :, column),
            view(gradients, :, column), field, column, dimension)
    end
end

"""
    WarmupHMC.rescore_warmup_checkpoint(lpdf, payload; rtol=1e-10)

Return an independent adaptive checkpoint with its current log density and all
retained pointwise gradients evaluated on `lpdf`. Use this explicitly before
resuming on another CPU; ordinary resume keeps its existing replay behavior.
Supply the same target, data and reparametrization structure as the original
run. Saved controls are applied privately, leaving the caller's controls alone.

Every old/new live log density and gradient component must be finite and obey
`abs(new - old) <= rtol * max(abs(old), 1)`. Retained positions and new log
densities must also be finite. `rtol` must be finite and nonnegative; zero
requires numerical equality. A mismatch throws `ArgumentError` with the
largest scaled deviation and its field, column and coordinate (zero denotes
the live scalar). The supplied payload is unchanged on success or failure.

Refreshes `position_and_gradient`, `posterior_gradient`, `halo_gradient`, and
occupied nonlinear leaf-sample gradients. Preserves RNGs, counters, window
phase, metric, dual averaging, accumulated moments and candidate losses.
Dropped-epoch snapshots are inert and remain in their original frame. This is
tolerance-checked continuation, not trajectory replay or proof of target
identity: compressed history cannot be reconstructed from retained points.

Appends an audit to `rescored`, preserved by subsequent checkpoints. Its
evaluation count is separate from sampler counters. Work is one target
evaluation per retained point plus O(dimension × retained points) vector work
and an independent payload copy; no dense adaptation matrix is constructed.
"""
function rescore_warmup_checkpoint(lpdf, payload::NamedTuple; rtol=1e-10)
    rtol isa Real && !(rtol isa Bool) && isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("rtol must be a finite nonnegative real number"))
    check_checkpoint_compatible(payload, :adaptive, (:adaptive,))
    _check_schema_version(payload)
    _check_candidate_scoring_compatible(payload, lpdf)
    p = deepcopy(payload)
    ir = _back_transform_reparametrizer(p, lpdf, p.dimension)
    target = _checkpoint_scoring_target(lpdf, ir)
    audit = CheckpointRescoreAudit()
    q = p.position_and_gradient
    value = _rescore_evaluate!(audit, target, q.q, q.∇ℓq,
        :position_and_gradient, 0, p.dimension)
    _rescore_compare!(audit, q.ℓq, value, :position_and_gradient, 0, 0)
    current = DynamicHMC.EvaluatedLogDensity(q.q, value, q.∇ℓq)
    _rescore_matrix!(audit, target, p.posterior_position, p.posterior_gradient,
        :posterior_gradient, p.dimension)
    _rescore_matrix!(audit, target, p.halo_position, p.halo_gradient,
        :halo_gradient, p.dimension)
    recorder = get(p, :nonlinear_recorder, nothing)
    sample = isnothing(recorder) ? nothing : _leaf_sample(recorder.online)
    if !isnothing(sample)
        isfinite(sample.total) && sample.total >= 0 || throw(ArgumentError(
            "checkpoint re-scoring: nonlinear leaf sample has invalid total weight"))
        if sample.total > 0
            _rescore_matrix!(audit, target, sample.position, sample.gradient,
                :nonlinear_sample_gradient, p.dimension)
        end
    end
    audit.max_scaled <= rtol || throw(ArgumentError(
        "checkpoint re-scoring exceeds rtol=$rtol: largest scaled deviation $(audit.max_scaled) at $(audit.location)"))
    history = NamedTuple[get(p, :rescored, NamedTuple[])...]
    push!(history, (; policy=:retained_pointwise, rtol,
        audit.n_evaluations, audit.changed_values, audit.changed_gradient_components,
        audit.max_abs, audit.max_scaled, audit.location,
        historical_state=:preserved))
    merge(p, (; position_and_gradient=current, rescored=history))
end
