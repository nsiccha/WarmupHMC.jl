# The shared harness behind the golden `adaptive_warmup_mcmc` baseline.
#
# Included by BOTH the `@testitem` in `golden_awm.jl` and the standalone
# `golden_awm_capture.jl`, so the runs being compared and the run being captured
# cannot drift apart. It defines no testsets and asserts nothing.
#
# A caller must already have the `targets.jl` targets (`DiagGaussian`, `Funnel`)
# and `AutoForwardDiff` in scope — the test item gets both from its
# `setup=[Targets, ADBackend]`, the capture script `include`s/`using`s them
# itself.

using WarmupHMC, Random, LogDensityProblems, LinearAlgebra, Serialization, Statistics
using WarmupHMC: reparam_sources, reparametrizer

const SEED = 20260718
const N_DRAWS = 200
const GOLDEN_PATH = joinpath(@__DIR__, "golden_awm.jls")
# `DiagGaussian` has a wide enough scale spread that the marginal-scale
# condition number exceeds `variance_cond_target` (2.0); the funnel's linear
# whitening cannot condition it at all. Both exercise the restart branch.
const TARGETS = (gaussian = DiagGaussian(exp.(range(-1.5, 1.5, 6))), funnel = Funnel(5))

# A REPARAMETRIZED funnel: exercises the state-in-lpdf path — `find_reparametrization!`
# mutates `reparametrizer.pairs` in place, so resume must restore those scalar
# source centerings onto a freshly-built problem. A fresh build below always
# starts from `source = PartiallyCentered(1.0)`; if the restore were a no-op,
# resumed windows would diverge.
reparam_funnel(k=5) = ReparametrizedProblem(
    IndexedReparametrization([
        i => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0), x -> 0.0, x -> x[1] / 2)
        for i in 2:(k + 1)
    ]),
    Funnel(k), AutoForwardDiff(),
)

function run_sampler(lpdf)
    rng = Xoshiro(SEED)
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=N_DRAWS, progress=nothing)
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
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=N_DRAWS, progress=nothing, callback=cb)
    (; result, rng_final=copy(rng), stages)
end

# Same run, opting into on-disk checkpoints. Result must equal the baseline, and
# the written checkpoints must deserialize cleanly.
function run_with_checkpoint(lpdf, dir)
    rng = Xoshiro(SEED)
    result = adaptive_warmup_mcmc(rng, lpdf; n_draws=N_DRAWS, progress=nothing, checkpoint_dir=dir)
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
canon_result(r) = merge(Base.structdiff(r, (; scale_options=nothing)),
                        (; selected=r.scale_options[r.active_transformation]))
canon(out) = map(out) do o
    (; result=canon_result(o.result), rng_final=o.rng_final)
end

# --- The checks, as data ----------------------------------------------------
# Returns a NamedTuple of named diff lists plus the raw runs, so the script mode
# can print them and the testset can assert on them without duplicating logic.
function golden_report()
    out = run_all()

    cbout = map(run_with_callback, TARGETS)
    dcb = diffs(canon(out), canon(cbout))

    # One directory PER TARGET. Since `0a4a7bc` a non-empty `checkpoint_dir`
    # without `resume`/`overwrite` is a hard error ("Refusing to guess"), so
    # sharing one directory across targets throws on the second — which is
    # exactly what this file did, undetected, for as long as it stayed unwired.
    dirs = map(_ -> mktempdir(), TARGETS)
    ckout = map(run_with_checkpoint, TARGETS, dirs)
    dck = diffs(canon(out), canon(ckout))
    cpfiles = sort(unique(reduce(vcat, [filter(f -> endswith(f, ".jls"), readdir(d))
                                        for d in dirs])))
    # The consumer-facing payload contract. `:stage` was in this list until
    # `0a4a7bc` made the payload pure sampler state and dropped it; the check
    # silently went red and stayed there. These are the keys a consumer that
    # renders a running fit actually reads.
    cpkeys = (:schema_version, :sampler, :dimension, :rng, :reparam_sources,
              :position_and_gradient, :posterior_position, :halo_position,
              :dropped_posterior_position, :n_divergent_samples, :n_samples,
              :outer_counter, :stepsize)
    cpok = !isempty(cpfiles) && all(dirs) do d
        fs = filter(f -> endswith(f, ".jls"), readdir(d))
        !isempty(fs) && all(fs) do f
            p = deserialize(joinpath(d, f))
            p isa NamedTuple && all(k -> haskey(p, k), cpkeys) && p.sampler === :adaptive
        end
    end

    # Resume ≡ straight-through: continuing from any written checkpoint must
    # reproduce the full straight-through result byte-for-byte.
    rdir = mktempdir()
    run_with_checkpoint(TARGETS.funnel, rdir)
    # `n_draws` MUST be re-passed. Since `0a4a7bc` config is no longer read from
    # the payload, so the wrapper's own default (1000) applies — comparing a
    # resumed 1000-draw run against a 200-draw straight-through run, which is
    # what this leg was silently doing.
    resume_diffs = map(["cp_init.jls", "cp_window_2.jls", "cp_latest.jls"]) do cp
        rr = resume_warmup_mcmc(TARGETS.funnel, joinpath(rdir, cp); n_draws=N_DRAWS, progress=nothing)
        cp => diffs(canon_result(out.funnel.result), canon_result(rr))
    end

    # Reparametrized resume: the state-in-lpdf path. A fresh problem starts at
    # source=1.0; restore must overwrite it with the checkpointed (optimized)
    # scalars for the continuation to match straight-through.
    rpdir = mktempdir()
    rpfull = run_with_checkpoint(reparam_funnel(), rpdir)
    rp_diffs = map(["cp_init.jls", "cp_latest.jls"]) do cp
        rr = resume_warmup_mcmc(reparam_funnel(), joinpath(rpdir, cp); n_draws=N_DRAWS, progress=nothing)
        cp => diffs(canon_result(rpfull.result), canon_result(rr))
    end

    # Frozen baseline, when one exists locally.
    golden = isfile(GOLDEN_PATH) ? deserialize(GOLDEN_PATH) : nothing
    dgolden = isnothing(golden) ? nothing : diffs(canon(golden), canon(out))

    (; out, cbout, ckout, dcb, dck, cpfiles, cpok, resume_diffs, rp_diffs,
       rpfull, rpdir, golden, dgolden)
end
