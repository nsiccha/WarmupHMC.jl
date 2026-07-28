# Golden harness for `adaptive_warmup_mcmc`.
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
# The frozen-baseline comparison is added on top whenever the file exists; write
# one with `julia --project=web/src/test web/src/test/golden_awm_capture.jl`.
#
# This is the slowest item in the suite. `--skip-tag=golden` drops it.

@testitem "the golden adaptive_warmup_mcmc harness" setup=[Targets, ADBackend, Determinism] tags=[:golden] begin
    include(joinpath(@__DIR__, "golden_awm_common.jl"))

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
                      "It is gitignored; run `julia --project=web/src/test " *
                      "web/src/test/golden_awm_capture.jl` to create one."
                @test true
            else
                isempty(r.dgolden) || foreach(d -> println("    ", d), first(r.dgolden, 15))
                @test isempty(r.dgolden)
            end
        end
    end
end
