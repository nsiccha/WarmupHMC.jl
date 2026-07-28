# Does `nonlinear_adapt=false` still return draws in the SAMPLER's frame, or does
# the sampler now back-transform them into the MODEL's frame itself?
#
# common.jl compensates for the former by calling `reparametrize!` on the fixed
# arms. `b109210` ("report draws in the model frame even when nonlinear_adapt=false")
# claims the latter. If it is right, the compensation now DOUBLE-transforms and
# every fixed-arm number in RESULTS.md would be wrong.
#
# The test cannot use an identity spec (target c == source c): identity applied
# twice is still identity, so it cannot tell the two apart. It needs a spec whose
# source and target genuinely differ, and a reference for what the answer should be
# -- which is the unwrapped sampler on the same target.
include(joinpath(@__DIR__, "common.jl"))
using Printf, Statistics
import JSON
# common.jl deliberately does NOT import this any more -- that it has no use for it
# is the property under test here. This probe needs it to build the wrong answer.
using WarmupHMC: reparametrize!

const SEED = 11
const NDRAWS = 400

f = Funnel(9)
dim = LogDensityProblems.dimension(f)

# Reference: the bare, unwrapped problem. Its draws are in the model frame by
# construction -- there is no wrapper to be in another frame of.
plain = Matrix{Float64}(adaptive_warmup_mcmc(Xoshiro(SEED), f; n_draws = NDRAWS,
                                             nonlinear_adapt = false).posterior_position)

# Wrapped, fixed, and genuinely transforming: model is centered (target 1.0), the
# sampler works noncentered (source 0.0).
spec = with_source(funnel_spec(f, 1.0), 0.0)
lpdf = ReparametrizedProblem(spec, f, AD_BACKEND)
res  = adaptive_warmup_mcmc(Xoshiro(SEED), lpdf; n_draws = NDRAWS, nonlinear_adapt = false)

as_returned  = Matrix{Float64}(res.posterior_position)
compensated  = copy(as_returned)
reparametrize!(lpdf, compensated)

summ(m) = (vec(mean(m, dims = 2)), vec(std(m, dims = 2)))
mp, sp = summ(plain)
ma, sa = summ(as_returned)
mc, sc = summ(compensated)

@printf("funnel d=%d, seed %d, %d draws (plain) / %d (wrapped)\n\n",
        dim, SEED, size(plain, 2), size(as_returned, 2))
@printf("%-5s | %19s | %19s | %19s\n", "coord", "plain (reference)",
        "as returned", "+ reparametrize!")
println(repeat("-", 74))
for i in 1:dim
    @printf("%-5d | %8.3f %8.3f  | %8.3f %8.3f  | %8.3f %8.3f\n",
            i, mp[i], sp[i], ma[i], sa[i], mc[i], sc[i])
end

# The funnel's legs have sd exp(x1/2); coordinate 1 has a known marginal, sd 3 for
# Funnel(9)'s standard form. Score each candidate by how far its per-coordinate sd
# sits from the reference, in units of the reference sd.
score(s) = maximum(abs.(s .- sp) ./ sp)
@printf("\nmax relative sd deviation from the unwrapped reference:\n")
@printf("  as returned      : %.3f\n", score(sa))
@printf("  + reparametrize! : %.3f\n", score(sc))
println()
const OUT_DIR = env_dir("WHMC_BENCH_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT_DIR)
already_model_frame = score(sa) < score(sc)
open(joinpath(OUT_DIR, "frame_check.json"), "w") do io
    JSON.print(io, Dict(
        "note" => "Whether nonlinear_adapt=false returns draws in the model frame " *
                  "(sampler back-transforms) or the source frame (harness must). " *
                  "common.jl assumes the former from b109210 onward.",
        "julia" => string(VERSION),
        git_provenance()...,
        "seed" => SEED, "n_draws" => NDRAWS, "dimension" => dim,
        "mean_plain" => mp, "sd_plain" => sp,
        "mean_as_returned" => ma, "sd_as_returned" => sa,
        "mean_compensated" => mc, "sd_compensated" => sc,
        "score_as_returned" => score(sa), "score_compensated" => score(sc),
        "already_model_frame" => already_model_frame), 2)
end
println("wrote $(joinpath(OUT_DIR, "frame_check.json"))\n")

if already_model_frame
    println("PASS: draws come back ALREADY in the model frame, which is what common.jl")
    println("      assumes. It applies no transform of its own. Nothing to do.")
else
    println("""
    FAIL: draws come back in the SAMPLER frame -- the pre-`b109210` behaviour.
          common.jl no longer compensates for that, so every fixed-`c` arm is
          currently scored on draws that were never mapped back into the model's
          frame. That is wrong in the direction that flatters the adaptive arm.
          Restore the single `reparametrize!(lpdf, draws)` on the fixed arms in
          `run_arm`, and re-run anything already measured on this base.""")
    exit(1)
end
