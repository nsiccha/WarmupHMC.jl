# Byte-identity golden harness for `adaptive_warmup_mcmc`.
#
# Freezes a baseline (returned NamedTuple + final rng state) from a fully
# deterministic run on a fixed, ill-conditioned diagonal-Gaussian target, then
# re-checks a fresh run against that frozen baseline under exact `==`.
#
#   julia --project test/golden_awm.jl capture   # write the baseline
#   julia --project test/golden_awm.jl check      # compare a fresh run to it
#
# Used to guarantee the checkpoint/resume + callbacks refactor stays byte-for-byte
# identical on the default path (see todo 2026-07-18T20-22-43-442-1mgraol).

using WarmupHMC, Random, LogDensityProblems, LinearAlgebra, Serialization
using ForwardDiff  # Pathfinder's default AutoForwardDiff needs it loaded in-session

# Pin BLAS to one thread so the baseline is reproducible: the adaptive
# transformation update runs multithreaded BLAS whose reduction order is
# non-deterministic run-to-run. Byte-identity is verified under this pin.
BLAS.set_num_threads(1)

# --- A fixed, self-contained target (no BridgeStan, pure Julia) -------------
# Diagonal Gaussian with a wide scale spread so the marginal-scale condition
# number exceeds `variance_cond_target` (2.0) and the outer loop restarts at
# least once — exercising the transformation-update branch (awm.jl:331-350).
struct DiagGaussian{V}
    sigma::V
end
LogDensityProblems.dimension(g::DiagGaussian) = length(g.sigma)
LogDensityProblems.capabilities(::Type{<:DiagGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(g::DiagGaussian, x) = -sum(abs2, x ./ g.sigma) / 2
LogDensityProblems.logdensity_and_gradient(g::DiagGaussian, x) =
    (LogDensityProblems.logdensity(g, x), -x ./ g.sigma .^ 2)

# Neal's funnel — v ~ N(0,3), xᵢ ~ N(0, exp(v/2)). Pathfinder's linear whitening
# cannot condition it, so `variance_cond` stays > 2 and the outer loop RESTARTS,
# exercising the transformation-update branch (awm.jl:327-350).
struct Funnel
    k::Int
end
LogDensityProblems.dimension(f::Funnel) = f.k + 1
LogDensityProblems.capabilities(::Type{Funnel}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(f::Funnel, x)
    v = x[1]; xs = @view x[2:end]
    -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
end
function LogDensityProblems.logdensity_and_gradient(f::Funnel, x)
    v = x[1]; xs = @view x[2:end]
    lp = -0.5 * v^2 / 9 - sum(xi -> 0.5 * xi^2 * exp(-v) + v / 2, xs)
    gv = -v / 9 + 0.5 * exp(-v) * sum(abs2, xs) - f.k / 2
    g = similar(x)
    g[1] = gv
    g[2:end] .= .-xs .* exp(-v)
    (lp, g)
end

const SEED = 20260718
const GOLDEN_PATH = joinpath(@__DIR__, "golden_awm.jls")
const TARGETS = (gaussian = DiagGaussian(exp.(range(-1.5, 1.5, 6))), funnel = Funnel(5))

function run_sampler(lpdf)
    rng = Xoshiro(SEED)
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=200, progress=nothing)
    (; result, rng_final=copy(rng))
end

run_all() = map(run_sampler, TARGETS)

# Same run, but with an OBSERVATIONAL callback (records stages, reads state, but
# consumes no rng and mutates nothing) — its result must equal the no-callback
# baseline, proving the checkpoint hook itself introduces no drift.
function run_with_callback(lpdf)
    rng = Xoshiro(SEED)
    stages = Symbol[]
    cb = (state, stage) -> (push!(stages, stage); state.outer_counter; nothing)
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=200, progress=nothing, callback=cb)
    (; result, rng_final=copy(rng), stages)
end

# Same run, opting into on-disk checkpoints. Result must equal the baseline, and
# the written checkpoints must deserialize cleanly.
function run_with_checkpoint(lpdf, dir)
    rng = Xoshiro(SEED)
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=200, progress=nothing, checkpoint_dir=dir)
    (; result, rng_final=copy(rng))
end

# --- Recursive exact comparison ---------------------------------------------
# Returns a list of dotted-path diffs (empty ⇒ byte-identical). Short-circuits on
# `isequal` (structural for the matrix-free AbstractMatrixExpression types, per
# 3b3c05d) so equal branches never index a matrix-free operator; only unequal
# nodes recurse, and the array branch is guarded by a getindex-capability check.
# Matrix-free WarmupHMC operators define no getindex (they throw CanonicalIndexError),
# yet inherit the AbstractArray linear-index fallback, so hasmethod/applicable can't
# see the gap — exclude the supertype explicitly and recurse them by fields instead.
indexable(a) = !(a isa WarmupHMC.AbstractMatrixExpression)
function diffs(a, b, path="")
    isequal(a, b) && return String[]
    if a isa AbstractArray && b isa AbstractArray && indexable(a) && indexable(b)
        size(a) != size(b) && return ["$path: size $(size(a)) != $(size(b))"]
        n = count(!isequal(x, y) for (x, y) in zip(a, b))
        return ["$path: $n/$(length(a)) elements differ"]
    elseif a isa NamedTuple && b isa NamedTuple && keys(a) == keys(b)
        return reduce(vcat, [diffs(a[k], b[k], "$path.$k") for k in keys(a)]; init=String[])
    elseif typeof(a) == typeof(b) && fieldcount(typeof(a)) > 0
        return reduce(vcat, [diffs(getfield(a, k), getfield(b, k), "$path.$k")
                             for k in fieldnames(typeof(a))]; init=String[])
    else
        return ["$path: $(repr(a)) != $(repr(b))"]
    end
end

# Canonicalize for comparison: the returned `scale_options` includes candidate
# operators that were NEVER selected/updated (e.g. the `adaptive`
# SuccessiveReflections when the target never restarts), whose s1/s2/loss buffers
# are `undef` (MatrixExpressions.jl:83-85) and so differ run-to-run. Compare only
# the SELECTED operator (always initialized) plus every substantive output.
canon(out) = map(out) do o
    r = o.result
    selected = r.scale_options[r.active_transformation]
    (; result=merge(Base.structdiff(r, (; scale_options=nothing)), (; selected)),
       rng_final=o.rng_final)
end

function main(mode)
    out = run_all()
    if mode == "capture"
        serialize(GOLDEN_PATH, out)
        println("CAPTURED baseline → $GOLDEN_PATH")
        for (name, o) in pairs(out)
            println("  [$name] draws=", size(o.result.posterior_position),
                    " active=", o.result.active_transformation,
                    " restarts=", length(o.result.scale_changes))
        end
    elseif mode == "check"
        golden = deserialize(GOLDEN_PATH)
        d = diffs(canon(golden), canon(out))
        # Observational callback must not perturb the result vs the baseline.
        cbout = map(run_with_callback, TARGETS)
        dcb = diffs(canon(golden), canon(cbout))
        for (name, o) in pairs(cbout)
            println("  [$name] callback stages: :init×$(count(==(:init), o.stages)) " *
                    ":window×$(count(==(:window), o.stages))")
        end
        # Opt-in disk checkpointing must not perturb the result, and every written
        # checkpoint must deserialize cleanly.
        dir = mktempdir()
        ckout = map(lpdf -> run_with_checkpoint(lpdf, dir), TARGETS)
        dck = diffs(canon(golden), canon(ckout))
        cpfiles = filter(f -> endswith(f, ".jls"), readdir(dir))
        cpok = all(cpfiles) do f
            p = deserialize(joinpath(dir, f))
            p isa NamedTuple && haskey(p, :stage) && haskey(p, :rng) && haskey(p, :reparam_sources)
        end
        println("  checkpoint files written: ", join(sort(cpfiles), ", "))
        println("  checkpoints deserialize OK: ", cpok)
        if isempty(d) && isempty(dcb) && isempty(dck) && cpok
            println("PASS — default, callback, and checkpoint paths byte-identical; checkpoints valid")
        else
            isempty(d) || (println("FAIL default — $(length(d)) diff(s):"); foreach(x -> println("  ", x), first(d, 15)))
            isempty(dcb) || (println("FAIL callback — $(length(dcb)) diff(s):"); foreach(x -> println("  ", x), first(dcb, 15)))
            isempty(dck) || (println("FAIL checkpoint — $(length(dck)) diff(s):"); foreach(x -> println("  ", x), first(dck, 15)))
            cpok || println("FAIL — a checkpoint failed to deserialize")
        end
    else
        error("usage: golden_awm.jl [capture|check]")
    end
end

main(get(ARGS, 1, "check"))
