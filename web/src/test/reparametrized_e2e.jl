@testitem "reparametrized end to end" setup=[Targets, ADBackend, Determinism] tags=[:reparametrization] begin
    using WarmupHMC, Random, LinearAlgebra, LogDensityProblems, Statistics
    using WarmupHMC: reparam_sources

    # The model-specific reparametrization factories the sampler is meant to be
    # driven with. Reusing them rather than hand-rolling means this item also
    # guards the specs themselves.
    include(joinpath(@__DIR__, "..", "posteriordb_reparametrizations.jl"))

    # END-TO-END guards: does `adaptive_warmup_mcmc` on a REPARAMETRIZED hierarchical
    # target recover the known posterior, and does adaptation move the centering
    # parameter somewhere sensible?
    #
    # TRAP THIS FILE EXISTS TO AVOID: `reparametrizer(::Any) = IndexedReparametrization([])`,
    # so `nonlinear_adapt=true` on a plain lpdf does NOTHING. Every test below
    # asserts that the centering actually MOVED before drawing any conclusion from
    # the posterior — otherwise a "reparametrized" run that silently degraded to the
    # plain sampler would pass as a success.

    BLAS.set_num_threads(1)

    const E2E_DRAWS = 2000

    centerings(rp) = [s.c for (_, s) in reparam_sources(rp)]

    # Neal's funnel, coordinate 1 = v, coordinates 2:k+1 = xᵢ.
    funnel_rp(k) = ReparametrizedProblem(
        IndexedReparametrization([
            (i + 1) => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(1.0),
                                         0.0, x -> x[1] / 2)
            for i in 1:k
        ]),
        Funnel(k), AutoForwardDiff(),
    )

    # Centered eight schools wrapped in the PosteriorDB spec (which starts every
    # coordinate at c = 1.0, i.e. fully centered).
    eight_schools_rp() = ReparametrizedProblem(
        reparametrization("eight_schools-eight_schools_centered", 10, nothing),
        EightSchools(true), AutoForwardDiff(),
    )

    @testset "reparametrized end to end" begin

        @testset "the PosteriorDB specs build a non-empty reparametrizer" begin
            # Trap 1, at the spec seam. If these are empty, every test below is
            # measuring the plain sampler.
            rp = eight_schools_rp()
            @test length(WarmupHMC.reparametrizer(rp).pairs) == 8
            @test centerings(rp) == fill(1.0, 8)         # starts fully centered
            @test WarmupHMC.reparametrizer(rp) !== WarmupHMC.reparametrizer(eight_schools_rp())
            # A bare target has none, and the wrapper must not change the density.
            @test isempty(WarmupHMC.reparametrizer(EightSchools(true)).pairs)
            x = randn(Xoshiro(1), 10)
            @test LogDensityProblems.logdensity(rp, x) ≈
                  LogDensityProblems.logdensity(EightSchools(true), x)
        end

        # --- Funnel: a target with an EXACTLY known marginal ---------------------
        @testset "funnel recovers v ~ N(0, 3) when adaptation fires" begin
            seeds = 1:6
            rows = map(seeds) do seed
                rp = funnel_rp(5)
                res = adaptive_warmup_mcmc(Xoshiro(seed), rp; n_draws=E2E_DRAWS, progress=nothing)
                plain = adaptive_warmup_mcmc(Xoshiro(seed), Funnel(5); n_draws=E2E_DRAWS, progress=nothing)
                c = centerings(rp)
                v = res.posterior_position[1, :]
                pv = plain.posterior_position[1, :]
                (; seed, c, adapted = any(!=(1.0), c),
                   sd = std(v), mean = mean(v), div = res.n_divergent_samples,
                   plain_sd = std(pv), plain_div = plain.n_divergent_samples)
            end
            println("  funnel (truth: mean(v)=0, sd(v)=3):")
            for r in rows
                println("    seed=$(r.seed) adapted=$(rpad(r.adapted, 5)) c=$(unique(r.c))  " *
                        "reparam sd(v)=$(round(r.sd, digits=2)) div=$(lpad(r.div, 3))  |  " *
                        "plain sd(v)=$(round(r.plain_sd, digits=2)) div=$(lpad(r.plain_div, 3))")
            end

            adapted = filter(r -> r.adapted, rows)
            n_adapted = length(adapted)
            println("  adaptation fired on $n_adapted/$(length(rows)) seeds")

            # EVERY seed must adapt. This was `>= 3` while the halo regression was
            # live: `variance_cond` is computed over the halo columns, so a starved
            # pool made the restart decision — the gate `find_reparametrization!`
            # sits behind — noisy, and seeds 3 and 4 never restarted on any window.
            # `095efb0` restored the recording rate and all six now fire. Keep this
            # at `== length(rows)`: a drop back to intermittent firing is the exact
            # symptom that regression produced, and it is invisible in the medians.
            @test n_adapted == length(rows)

            @testset "adapted runs recover the known marginal" begin
                for r in adapted
                    # Truth is sd(v) = 3 exactly. A centered funnel cannot reach the
                    # neck and reports a value biased LOW; this is the check that the
                    # reparametrization actually fixed the geometry.
                    @test 2.6 < r.sd < 3.5
                    @test abs(r.mean) < 0.6
                    # Measured 0 on all six seeds; the bound is headroom for
                    # platform/BLAS variation, not a claim that a few are expected.
                    @test r.div <= 5
                    # It should have moved toward non-centered, which is the right
                    # answer for a funnel with no data.
                    @test all(<=(0.5), r.c)
                end
            end

            @testset "adapting beats not adapting" begin
                # Same seeds, same target, with and without the reparametrization.
                # The adapted runs must be closer to sd(v) = 3 than their plain
                # counterparts, and must diverge less.
                err_reparam = median([abs(r.sd - 3) for r in adapted])
                err_plain = median([abs(r.plain_sd - 3) for r in adapted])
                println("  median |sd(v) - 3|: reparam=$(round(err_reparam, digits=3)) " *
                        "plain=$(round(err_plain, digits=3))")
                @test err_reparam < err_plain
                @test sum(r.div for r in adapted) < sum(r.plain_div for r in adapted)
            end
        end

        # --- Eight schools: the canonical demonstration --------------------------
        @testset "centered eight_schools adapts toward non-centered" begin
            seeds = 1:5
            rows = map(seeds) do seed
                rp = eight_schools_rp()
                rep = adaptive_warmup_mcmc(Xoshiro(seed), rp; n_draws=E2E_DRAWS, progress=nothing)
                cen = adaptive_warmup_mcmc(Xoshiro(seed), EightSchools(true); n_draws=E2E_DRAWS, progress=nothing)
                non = adaptive_warmup_mcmc(Xoshiro(seed), EightSchools(false); n_draws=E2E_DRAWS, progress=nothing)
                lt(r) = r.posterior_position[10, :]      # log τ — the funnel-shaped coordinate
                (; seed, c = centerings(rp),
                   rep_m = mean(lt(rep)), rep_s = std(lt(rep)), rep_d = rep.n_divergent_samples,
                   cen_m = mean(lt(cen)), cen_s = std(lt(cen)), cen_d = cen.n_divergent_samples,
                   non_m = mean(lt(non)), non_s = std(lt(non)), non_d = non.n_divergent_samples)
            end
            println("  eight_schools log τ (reference = the non-centered parametrization):")
            for r in rows
                println("    seed=$(r.seed) c=$(unique(r.c))")
                println("      reparam(from centered) mean=$(round(r.rep_m, digits=3)) sd=$(round(r.rep_s, digits=3)) div=$(r.rep_d)")
                println("      plain centered         mean=$(round(r.cen_m, digits=3)) sd=$(round(r.cen_s, digits=3)) div=$(r.cen_d)")
                println("      plain non-centered     mean=$(round(r.non_m, digits=3)) sd=$(round(r.non_s, digits=3)) div=$(r.non_d)")
            end

            @testset "the centering moves off fully-centered on every seed" begin
                # THE demonstration: started at c = 1.0 for all eight coordinates,
                # adaptation must discover that this target wants non-centering.
                for r in rows
                    @test any(!=(1.0), r.c)
                    @test maximum(r.c) <= 0.5
                end
            end

            @testset "it recovers the non-centered reference posterior" begin
                # The non-centered parametrization samples this target cleanly, so
                # its log τ marginal is the reference. Compared across seeds (median,
                # so one unlucky seed cannot decide the outcome), the reparametrized
                # run from a CENTERED start must land closer to that reference than
                # the plain centered run does — in both location and spread.
                ref_m = median([r.non_m for r in rows])
                ref_s = median([r.non_s for r in rows])
                rep_m, rep_s = median([r.rep_m for r in rows]), median([r.rep_s for r in rows])
                cen_m, cen_s = median([r.cen_m for r in rows]), median([r.cen_s for r in rows])
                println("  median log τ — reference (mean, sd) = ($(round(ref_m, digits=3)), $(round(ref_s, digits=3)))")
                println("    reparam  = ($(round(rep_m, digits=3)), $(round(rep_s, digits=3)))")
                println("    centered = ($(round(cen_m, digits=3)), $(round(cen_s, digits=3)))")
                @test abs(rep_m - ref_m) < abs(cen_m - ref_m)
                @test abs(rep_s - ref_s) < abs(cen_s - ref_s)
                # And in absolute terms, close to the reference rather than merely
                # less wrong than a badly-biased baseline.
                @test abs(rep_m - ref_m) < 0.15
                @test abs(rep_s - ref_s) < 0.15
            end

            @testset "it does not diverge more than the non-centered reference" begin
                # The plain centered run is the one that should be divergence-prone.
                @test median([r.rep_d for r in rows]) <= median([r.cen_d for r in rows])
            end
        end
    end
end
