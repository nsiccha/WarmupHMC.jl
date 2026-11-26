# Progress functions come from Treebars (using Treebars in WarmupHMC.jl).
# This file only defines domain-specific display types and short_string/round2 extensions.

Treebars.round2(x::OnlineStatsBase.Mean) = Treebars.round2(mean(x))

struct UncertainFrequency
    obs::Int64
    n::Int64
end
Base.string(uf::UncertainFrequency) = "$(uf.obs) out of $(uf.n) ($(short_string(100*quantile(Beta(1+uf.obs, 1+uf.n-uf.obs), .05))) - $(short_string(100*quantile(Beta(1+uf.obs, 1+uf.n-uf.obs), .95)))%)"

struct SamplingPerformance
    stepsize::Float64
    steps::Union{Nothing,Float64}
    unit::String
end
function Base.string(x::SamplingPerformance)
    rate = isnothing(x.steps) ? "awaiting $(x.unit)" : "$(short_string(x.steps)) steps per $(x.unit)"
    "$rate (stepsize = $(short_string(x.stepsize)))"
end
_sampling_performance(chain) = SamplingPerformance(chain.stepsize, mean(chain.steps_per_draw), "draw")
function _sampling_performance(state::AWMState)
    if state.n_samples > 0
        n = state.n_samples - state.uncosted_retained_draws
        return SamplingPerformance(state.stepsize, n > 0 ? state.sampling_evaluation_counter / n : nothing,
                                   "retained draw")
    end
    if OnlineStatsBase.nobs(state.steps_per_draw) > 0
        return SamplingPerformance(state.stepsize, mean(state.steps_per_draw), "tuning transition")
    end
    unit = state.current_transition_counter < state.stepsize_adaptation_limit ? "tuning transition" : "retained draw"
    SamplingPerformance(state.stepsize, nothing, unit)
end

struct ActiveTransformation{K}
    kinetic_energy::K
    scale_changes::Vector{Float64}
end

struct Speed
    n::Int
    dt::Float64
    Speed(n, dt::UInt64) = new(n, Float64(dt))
end
# Render a throughput. A per-second RATE reads badly for a SLOW process
# (`0.016 / s`), so once it drops below 1/s we show the reciprocal PACE — the
# time PER event — via Treebars' `short_duration` (`pace: 1m 4s`, `pace: 2h 49m`,
# `pace: 11d 13h`). FAST/large rates already pretty-print through `short_string`
# (`5k / s`), so those stay as a rate. `n` events over `seconds` wall-clock.
_rate_or_pace(n::Real, seconds::Real) = begin
    rate = n / seconds
    (isfinite(rate) && 0 < rate < 1) ?
        "pace: $(short_duration(Dates.Millisecond(round(Int, 1000 / rate))))" :
        "$(short_string(rate)) / s"
end
Base.string(x::Speed) = "$(x.n) in $(short_string(x.dt/1e9)) seconds ($(_rate_or_pace(x.n, x.dt/1e9)))"
Base.string(x::ActiveTransformation) = "$(short_string(x.kinetic_energy.M⁻¹.m1)) (marginal scale changes = $(short_string(_ScaleChange.(x.scale_changes))))"
# One entry of `scale_changes`, as the label shows it. A restart test that could
# not measure a coordinate's scale, because the window's recorded positions never
# moved there (or its gradient never changed), has no finite factor: it reads
# `frozen`, never `Inf` (snag collapsed-chains-10d2df94). The returned and
# checkpointed `scale_changes` keep the number.
struct _ScaleChange
    x::Float64
end
Treebars.short_string(c::_ScaleChange) = isfinite(c.x) ? short_string(c.x) : "frozen"

Treebars.short_string(x::WarmupHMC.MatrixFactorization{<:Any, <:LinearAlgebra.Transpose}) = short_string(parent(x.m1))
# `_initial_pathfinder_scale` takes a DENSE Cholesky factor only for a
# caller-supplied dense squared scale (a Pathfinder init keeps the low-rank
# factor and renders as `Pathfinder(k)` above). From `f210206` until the
# low-rank factor was restored, EVERY Pathfinder init was dense, and this label
# fell through to the generic `string` of a d×d matrix: ~43k characters at
# d = 50, ~2.7M at d = 400, rebuilt on every progress render.
Treebars.short_string(x::WarmupHMC.MatrixFactorization{<:Any, <:LinearAlgebra.LowerTriangular}) =
    "Pathfinder(dense, d = $(size(x.m1, 1)))"
# Any other factor: name it, never print it. A progress label must stay O(1) in
# the dimension whatever factor a future init path produces.
Treebars.short_string(x::WarmupHMC.MatrixFactorization) =
    "$(nameof(typeof(x.m1)))(d = $(size(x.m1, 1)))"
Treebars.short_string(x::Pathfinder.WoodburyPDRightFactor) = "Pathfinder($(size(x.V, 1)))"
Treebars.short_string(x::WarmupHMC.MatrixFactorization{<:Any, <:WarmupHMC.SuccessiveReflections}) = "Adaptive($(length(x.m1.reflections)))"
Treebars.short_string(x::Diagonal) = "Diagonal($(short_string(diag(x))))"
Treebars.short_string(x::Beta) = "$(short_string(x.α)) out of $(short_string(x.α+x.β)) ($(short_string(100*quantile(x, .05))) - $(short_string(100*quantile(x, .95)))%)"

# The labelled children of one adaptive-style chain node (warmuphmc-use §5b).
# `adaptive_warmup_mcmc` (`AWMState`) and `cooperative_warmup_mcmc`
# (`CooperativeChain`) both call these: the two structs hold the same counters
# under the same field names. `counter` is the node's bar value, or `nothing`
# for a node without a bar. A `nothing` node skips building the labels.

# Before the first transition.
_initial_chain_progress!(::Nothing, counter, chain) = nothing
_initial_chain_progress!(progress, counter, chain) = update_progress!(progress, counter;
    divergent_samples=UncertainFrequency(chain.n_divergent_samples, chain.n_samples),
    (chain.monitor_ess ? (;ess="pending...") : (;))...,
    active_transformation=ActiveTransformation(chain.kinetic_energy, chain.scale_changes),
    sampling_performance=_sampling_performance(chain),
    chain.total_transition_counter,
    chain.total_evaluation_counter,
)
# After every transition.
_transition_progress!(::Nothing, counter, chain) = nothing
_transition_progress!(progress, counter, chain) = update_progress!(progress, counter;
    divergent_samples=UncertainFrequency(chain.n_divergent_samples, chain.n_samples),
    # Only once the target has answered +Inf or NaN (`_take_nonfinite!`), so a
    # healthy chain gets no nonfinite-leaf label.
    (chain.n_nonfinite_leaves > 0 ? (; nonfinite_leaves=
        "$(chain.n_nonfinite_leaves) leapfrog points where the target answered +Inf or NaN (counted as divergences)") : (;))...,
    active_transformation=ActiveTransformation(chain.kinetic_energy, chain.scale_changes),
    sampling_performance=_sampling_performance(chain),
    total_transition_counter=Speed(chain.total_transition_counter, time_ns()-chain.start_time),
    total_evaluation_counter=Speed(chain.total_evaluation_counter, time_ns()-chain.start_time),
)
# At the end of a window that retained more than 10 draws.
_ess_progress!(::Nothing, chain) = nothing
_ess_progress!(progress, chain) = update_progress!(progress, nothing;
    ess=_ess_label(chain.ess) * " from $(chain.n_samples) samples.",
)
# Explain a backwards step on the chain node before publishing the lower count.
_reset_chain_progress!(::Nothing, state, event) = nothing
function _reset_chain_progress!(progress, state, event)
    update_progress!(progress, "$event after window $(state.outer_counter)")
    update_progress!(progress, state.current_transition_counter; sampling_performance=_sampling_performance(state))
end
# After a restart reselected the linear transformation.
_transformation_progress!(::Nothing, chain) = nothing
_transformation_progress!(progress, chain) = update_progress!(progress, nothing;
    active_transformation=ActiveTransformation(chain.kinetic_energy, chain.scale_changes),
)

# The diagonal entries of L in its own axes. A change in the outer rotation or
# nonlinear controls makes entrywise ratios to the previous metric meaningless.
_axis_scales(L::Diagonal) = L.diag
_axis_scales(L::MatrixFactorization) = L.m2.diag
_fit_axes(L) = nothing
_fit_axes(L::ScaleThenReflect) = (L.m1.idxs, L.m1.reflections)

function _axis_extrema(values)
    bad = findfirst(x -> !(isfinite(x) && x > 0), values)
    isnothing(bad) || return "unmeasurable at axis $bad"
    lo, i = findmin(values)
    hi, j = findmax(values)
    "min $(short_string(lo)) at axis $i, max $(short_string(hi)) at axis $j"
end

_refit_snapshot(::Nothing, state, reached_budget) = nothing
function _refit_snapshot(progress, state, reached_budget)
    active = state.active_transformation
    scale = state.scale_options[active]
    (; active, scales=copy(_axis_scales(scale)), axes=deepcopy(_fit_axes(scale)),
       incoming=reached_budget ? _axis_extrema(state.variance_memory) : nothing,
       limits=map(L -> falses(size(L, 1)), state.scale_options))
end

_refit_progress!(::Nothing, state, losses, nonlinear_changed, fallback) = nothing
function _refit_progress!(report, state, losses, nonlinear_changed, fallback)
    selected = state.active_transformation
    scale = state.scale_options[selected]
    same_axes = selected === report.active && !nonlinear_changed && _fit_axes(scale) == report.axes
    installed = _axis_scales(scale)
    range = same_axes ? "scale multipliers " * _axis_extrema(installed ./ report.scales) :
                        "new-frame scales " * _axis_extrema(installed)
    scores = join(("$name $(round(loss; sigdigits=6))" for (name, loss) in pairs(losses)), ", ")
    choice = if !isfinite(losses[selected])
        " (no finite loss)"
    elseif count(==(losses[selected]), losses) > 1
        " (tied minimum)"
    else
        ""
    end
    incoming = isnothing(report.incoming) ? "" : " | incoming corrections $(report.incoming)"
    reflections = length(state.scale_options.adaptive.m1.reflections)
    reflection_label = reflections == 1 ? "1 reflection" : "$reflections reflections"
    limits = "one-sided limits $(count(report.limits[selected]))"
    fallback && (limits *= " in fit, streaming correction installed")
    update_progress!(state.progress, nothing; metric_refit=
        "window $(state.outer_counter), $(report.active) → $selected$choice | losses $scores ($reflection_label)" *
        incoming * " | $range | $limits")
end

struct Speeds
    n::Vector{Int}
    dt::Float64
    Speeds(n, dt::UInt64) = new(n, Float64(dt))
end
@views Base.string(x::Speeds) = begin
    p = sortperm(x.n)
    dt = (time_ns() - x.dt)/1e9
    "$(short_string(x.n[p])) in $(short_string(dt)) seconds ($(short_string(x.n[p]./(dt))) / s)"
end

struct Speeds2{T}
    n::T
    dt::Float64
    Speeds2(n, dt::UInt64) = new{typeof(n)}(n, Float64(dt))
end
@views Base.string(x::Speeds2) = begin
    p = sortperm(x.n)
    dt = (time_ns() - x.dt)/1e9
    "$(short_string(x.n[p])) in $(short_string(dt)) seconds ($(short_string(x.n[p]./(dt))) / s)"
end
