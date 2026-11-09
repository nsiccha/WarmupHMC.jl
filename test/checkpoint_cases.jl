# Shared with fresh-process and cross-revision checkpoint acceptance runs.
module AdaptiveCheckpointCases
using WarmupHMC, Random, LogDensityProblems, Serialization

struct Banana
    delay::Float64
end
Banana() = Banana(0.0)
LogDensityProblems.dimension(::Banana) = 2
LogDensityProblems.capabilities(::Type{Banana}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::Banana, x)
    p.delay > 0 && sleep(p.delay)
    z = x[2] - 0.4 * (x[1]^2 - 1)
    (-x[1]^2 / 2 - z^2 / 2, [-x[1] + 0.8 * x[1] * z, -z])
end
LogDensityProblems.logdensity(p::Banana, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

struct WatchingBanana{F}
    observe::F
end
LogDensityProblems.dimension(::WatchingBanana) = 2
LogDensityProblems.capabilities(::Type{<:WatchingBanana}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::WatchingBanana, x)
    p.observe()
    LogDensityProblems.logdensity_and_gradient(Banana(), x)
end
LogDensityProblems.logdensity(p::WatchingBanana, x) = first(LogDensityProblems.logdensity_and_gradient(p, x))

const CONFIG = (; n_draws=40, n_evaluations=10_000, stepsize_adaptation_limit=8,
    recording_target=32, max_tree_depth=5, monitor_ess=true,
    init=(; position=[0.4, -0.2], squared_scale=ones(2)))
const RESTART_CONFIG = merge(CONFIG, (; n_evaluations=16, variance_cond_target=1.0))

function stop_after(n)
    calls = Ref(0)
    () -> (calls[] += 1; calls[] >= n)
end

latest(dir) = deserialize(joinpath(dir, "cp_latest.jls"))

# Ignore inactive metric candidates' uninitialized scratch buffers. Compare the
# learned active operator as its documented data representation, and every other
# persisted state field recursively (not just returned draw counts).
function comparable(p)
    merge(Base.structdiff(p, (; scale_options=nothing)),
        (; active_scale=WarmupHMC.scale_data(p.scale_options[p.active_transformation])))
end
function same(a, b)
    isequal(a, b) && return true
    typeof(a) === typeof(b) || return false
    a isa AbstractArray && return size(a) == size(b) && all(same(x, y) for (x, y) in zip(a, b))
    fieldcount(typeof(a)) > 0 || return false
    all(same(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
end

function run_case(dir; resume=false, stop_at=nothing, config=CONFIG, kwargs...)
    adaptive_warmup_mcmc(Xoshiro(resume ? 999 : 123), Banana(); config...,
        checkpoint_dir=dir, resume,
        stop_requested=isnothing(stop_at) ? nothing : stop_after(stop_at), kwargs...)
end
end
