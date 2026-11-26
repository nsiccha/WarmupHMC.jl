# A slot owns its counter even when initialization or a transition throws.
# Only its worker writes to it. The coordinator reads it after the workers join.
mutable struct _CompletionCost
    calls::Base.RefValue{Int64}  # this invocation only
    prior_calls::Union{Int,Nothing}
    initialization_calls::Union{Int,Nothing}
    sampling_calls::Union{Int,Nothing}
    total_steps::Int
    sampling_steps::Union{Int,Nothing}
    windows::Int
end
_CompletionCost() = _CompletionCost(Ref(Int64(0)), 0, 0, 0, 0, 0, 0)
struct _CompletionTarget{P} <: WrappedLogDensityProblem{P}
    target::P
    cost::_CompletionCost
end
Base.parent(p::_CompletionTarget) = p.target
LogDensityProblems.logdensity_and_gradient(p::_CompletionTarget, x) =
    _counted_gradient(p.cost.calls, parent(p), x)
_completion_cost(p) = nothing
_completion_cost(p::_CompletionTarget) = p.cost
# Preserve the chain's original target on the public callback state. Sampler
# evaluations go through the counted parent of its trajectory recorder.
_callback_target(p) = p
_callback_target(p::_CompletionTarget) = parent(p)
_add_cost(a, b) = isnothing(a) || isnothing(b) ? nothing : a + b

# Seed even the slots the admission pre-pass may leave unstarted. Their
# checkpointed work belongs to the run, but never to this attempt's elapsed time.
function _restore_completion_cost!(c, payload)
    empty_epoch = iszero(size(payload.posterior_position, 2))
    c.prior_calls = get(payload, :gradient_calls, nothing)
    c.initialization_calls = get(payload, :initialization_gradient_calls, nothing)
    c.sampling_calls = get(payload, :sampling_gradient_calls, empty_epoch ? 0 : nothing)
    c.total_steps = payload.total_evaluation_counter
    c.sampling_steps = get(payload, :uncosted_retained_draws, 0) > 0 ? nothing :
        get(payload, :sampling_evaluation_counter, empty_epoch ? 0 : nothing)
    c.windows = payload.outer_counter
    nothing
end

_cost_fields(::Nothing) = (;)
_cost_fields(c::_CompletionCost) = (;
    gradient_calls=_add_cost(c.prior_calls, c.calls[]),
    initialization_gradient_calls=c.initialization_calls,
    sampling_gradient_calls=c.sampling_calls)
_slot_cost(c) = (; _cost_fields(c)..., attempt_gradient_calls=c.calls[],
    total_evaluation_counter=c.total_steps, sampling_evaluation_counter=c.sampling_steps,
    n_warmup_windows=c.windows)
const _COMPLETION_COST_FIELDS = keys(_slot_cost(_CompletionCost()))

_call_count(::Nothing) = 0
_call_count(c::_CompletionCost) = c.calls[]
_charge_draw!(::Nothing, calls) = nothing
_charge_draw!(c::_CompletionCost, calls) = (c.sampling_calls = _add_cost(c.sampling_calls, calls))
_reset_sampling_cost!(::Nothing) = nothing
_reset_sampling_cost!(c::_CompletionCost) = (c.sampling_calls = 0)

_measure_setup(f, ::Nothing, fresh) = f()
function _measure_setup(f, c::_CompletionCost, fresh)
    if fresh
        # A second centering pass starts a new sampling epoch, even if its
        # initializer fails before it can construct an adaptive state.
        _reset_sampling_cost!(c)
        c.sampling_steps = 0
    end
    before = c.calls[]
    try
        f()
    finally
        c.initialization_calls = _add_cost(c.initialization_calls, c.calls[] - before)
    end
end

_finish_pass!(::Nothing, state, fresh) = nothing
function _finish_pass!(c::_CompletionCost, state, fresh)
    # Fresh passes have their own zero-based native counters. A restored
    # state already includes the prefix initially read from its checkpoint.
    c.total_steps = (fresh ? c.total_steps : 0) + state.total_evaluation_counter
    c.windows = (fresh ? c.windows : 0) + state.outer_counter
    c.sampling_steps = state.uncosted_retained_draws > 0 ? nothing : state.sampling_evaluation_counter
end
