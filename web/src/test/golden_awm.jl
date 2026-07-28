# Golden harness for `adaptive_warmup_mcmc`.
#
#   julia --project test/golden_awm.jl capture   # write the frozen baseline
#   julia --project test/golden_awm.jl check     # compare a fresh run to it
#
# and, when `include`d from `runtests.jl`, the same checks as a `@testset`.
#
# WHAT IS AND IS NOT GUARDED HERE
#
# The frozen baseline (`golden_awm.jls`) is **gitignored and not committed** —
# see the `*.jls` rule in `.gitignore`. It is a LOCAL artifact, regenerated from
# whatever the working tree currently does, so it cannot by itself catch a
# cross-commit regression: capture after a behaviour change and it simply pins
# the new behaviour. That is not a defect to fix by committing it (the file is
# platform- and BLAS-sensitive), but it does mean the byte-identity leg is a
# *within-session determinism* check, not a correctness check.
#
# So the real guards are the ones that need no baseline, and they run always:
#
#   1. INTERNAL EQUIVALENCE — the default path, an observational callback, and
#      opt-in disk checkpointing must produce byte-identical results, and a
#      resume from any checkpoint must reproduce the straight-through run.
#   2. PROPERTY ASSERTIONS — invariants byte-identity provably cannot see.
#      A frozen baseline pins whatever the code produced, including an INVERTED
#      back-transform (the bug `reparametrize_direction.jl` guards): re-capture
#      and the wrong values become "correct". The properties below are stated
#      against the model's known truth instead.
#
# The frozen-baseline comparison is added on top whenever the file exists.

using WarmupHMC, Random, LogDensityProblems, LinearAlgebra, Serialization, Statistics, Test
using WarmupHMC: reparam_sources, reparametrizer

include(joinpath(@__DIR__, "ad_backend.jl"))
include(joinpath(@__DIR__, "targets.jl"))

# Pin BLAS to one thread so the run is reproducible: the adaptive
# transformation update runs multithreaded BLAS whose reduction order is
# non-deterministic run-to-run.
BLAS.set_num_threads(1)

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

function main(mode)
    if mode == "capture"
        out = run_all()
        serialize(GOLDEN_PATH, out)
        println("CAPTURED baseline → $GOLDEN_PATH")
        for (name, o) in pairs(out)
            println("  [$name] draws=", size(o.result.posterior_position),
                    " active=", o.result.active_transformation,
                    " restarts=", length(o.result.scale_changes))
        end
        return
    end
    mode == "check" || error("usage: golden_awm.jl [capture|check]")
    r = golden_report()
    for (name, o) in pairs(r.cbout)
        println("  [$name] callback stages: :init×$(count(==(:init), o.stages)) " *
                ":window×$(count(==(:window), o.stages))")
    end
    println("  checkpoint files written: ", join(sort(r.cpfiles), ", "))
    println("  checkpoints deserialize OK: ", r.cpok)
    for (cp, rd) in r.resume_diffs
        println("  resume from $cp: ", isempty(rd) ? "byte-identical" : "$(length(rd)) DIFF(s)")
        foreach(x -> println("      ", x), first(rd, 8))
    end
    for (cp, rd) in r.rp_diffs
        println("  reparam resume from $cp: ", isempty(rd) ? "byte-identical" : "$(length(rd)) DIFF(s)")
    end
    resume_ok = all(isempty(last(x)) for x in r.resume_diffs)
    reparam_ok = all(isempty(last(x)) for x in r.rp_diffs)
    if isnothing(r.dgolden)
        println("  frozen baseline: ABSENT at $GOLDEN_PATH (run `capture` to create one)")
    else
        println("  frozen baseline: ", isempty(r.dgolden) ? "byte-identical" : "$(length(r.dgolden)) DIFF(s)")
        foreach(x -> println("      ", x), first(something(r.dgolden, String[]), 15))
    end
    if isempty(r.dcb) && isempty(r.dck) && r.cpok && resume_ok && reparam_ok &&
       (isnothing(r.dgolden) || isempty(r.dgolden))
        println("PASS — default/callback/checkpoint byte-identical; checkpoints valid; resume ≡ straight-through")
    else
        isempty(r.dcb) || (println("FAIL callback — $(length(r.dcb)) diff(s):"); foreach(x -> println("  ", x), first(r.dcb, 15)))
        isempty(r.dck) || (println("FAIL checkpoint — $(length(r.dck)) diff(s):"); foreach(x -> println("  ", x), first(r.dck, 15)))
        r.cpok || println("FAIL — a checkpoint failed to deserialize")
        resume_ok || println("FAIL — resume diverged from straight-through")
        reparam_ok || println("FAIL — reparametrized resume diverged")
    end
end

# NB: `abspath` BOTH sides. `@__FILE__` is the path as Julia was handed it, so
# `julia --project web/src/test/golden_awm.jl` leaves it relative and the naive
# comparison silently falls through to the testset branch. And `using Test` has
# to be at top level (above), not inside this `if`: the whole `if` is one
# top-level expression, so `@testset` is macro-expanded before any branch runs.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(get(ARGS, 1, "check"))
else
    @testset "golden adaptive_warmup_mcmc" begin
        r = golden_report()

        @testset "an observational callback does not perturb the run" begin
            isempty(r.dcb) || foreach(d -> println("    ", d), first(r.dcb, 15))
            @test isempty(r.dcb)
            for (_, o) in pairs(r.cbout)
                @test count(==(:init), o.stages) == 1
                @test count(==(:window), o.stages) >= 1
            end
        end

        @testset "opt-in checkpointing does not perturb the run" begin
            isempty(r.dck) || foreach(d -> println("    ", d), first(r.dck, 15))
            @test isempty(r.dck)
            @test !isempty(r.cpfiles)
            @test r.cpok
            @test "cp_latest.jls" in r.cpfiles
            @test "cp_init.jls" in r.cpfiles
        end

        @testset "resume reproduces the straight-through run" begin
            for (cp, rd) in r.resume_diffs
                isempty(rd) || (println("    resume from $cp:"); foreach(d -> println("      ", d), first(rd, 8)))
                @test isempty(rd)
            end
        end

        @testset "reparametrized resume restores the adapted sources" begin
            for (cp, rd) in r.rp_diffs
                isempty(rd) || (println("    reparam resume from $cp:"); foreach(d -> println("      ", d), first(rd, 8)))
                @test isempty(rd)
            end
            # The restore is only load-bearing if adaptation actually MOVED the
            # sources away from the fresh-build value of 1.0. Without this the
            # testset above would pass vacuously — a no-op restore onto a fresh
            # problem that already matched would look identical.
            payload = deserialize(joinpath(r.rpdir, "cp_latest.jls"))
            checkpointed = [s.c for (_, s) in payload.reparam_sources]
            println("    checkpointed sources: ", checkpointed)
            @test !isempty(checkpointed)
            @test any(!=(1.0), checkpointed)
        end

        # --- Property assertions: what byte-identity cannot see ---------------
        @testset "returned draws are in MODEL coordinates" begin
            # An inverted back-transform (`inverse(ir)` instead of `ir`) leaves
            # the draws in SOURCE coordinates. A frozen baseline captured after
            # such a bug pins the wrong values as "correct"; this does not.
            #
            # Discriminating statistic: the funnel's model coordinates have
            # xᵢ ~ N(0, exp(v/2)), so log|xᵢ| rises with v. After adaptation the
            # source is non-centered, where x̃ᵢ ~ N(0,1) is INDEPENDENT of v.
            # Correlation near zero therefore means the draws were never mapped
            # back into the model's parametrization.
            rp = reparam_funnel()
            res = adaptive_warmup_mcmc(Xoshiro(SEED), rp; n_draws=2000, progress=nothing)
            sources = [s.c for (_, s) in reparam_sources(rp)]
            println("    adapted funnel sources: ", sources)
            @test all(!=(1.0), sources)      # otherwise source == target and the test is vacuous

            draws = res.posterior_position
            v = draws[1, :]
            cors = [cor(v, log.(abs.(draws[i, :]))) for i in 2:size(draws, 1)]
            println("    cor(v, log|xᵢ|) on returned draws: ", round.(cors, digits=3))
            # In model coordinates this is strongly positive (theoretically ~0.5·
            # sd(v)/sd(log|x|)); in source coordinates it is ~0.
            @test all(>(0.3), cors)

            # The sampler's own current position stays in SOURCE coordinates, so
            # with source != target the two frames must disagree.
            @test !(res.position_and_gradient.q ≈ draws[:, end])
        end

        @testset "a plain lpdf is untouched by the back-transform" begin
            # `nonlinear_adapt=true` on a bare target must be an exact no-op, so
            # the funnel run above and a `nonlinear_adapt=false` run must agree
            # byte-for-byte.
            a = adaptive_warmup_mcmc(Xoshiro(SEED), TARGETS.funnel; n_draws=N_DRAWS, progress=nothing)
            b = adaptive_warmup_mcmc(Xoshiro(SEED), TARGETS.funnel; n_draws=N_DRAWS, progress=nothing,
                                     nonlinear_adapt=false)
            @test isempty(diffs(canon_result(a), canon_result(b)))
        end

        @testset "frozen baseline (only if one exists locally)" begin
            if isnothing(r.dgolden)
                @info "golden_awm.jls absent — byte-identity vs a frozen baseline not checked. " *
                      "It is gitignored; run `julia --project test/golden_awm.jl capture` to create one."
                @test true
            else
                isempty(r.dgolden) || foreach(d -> println("    ", d), first(r.dgolden, 15))
                @test isempty(r.dgolden)
            end
        end
    end
end
