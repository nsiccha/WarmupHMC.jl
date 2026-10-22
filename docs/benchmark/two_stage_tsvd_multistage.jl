# Follow-up to `two_stage_tsvd_fit.jl`: how many reflection + diagonal stages
# keep adding reflections, and what does each stage buy?
#
#   julia --project=. docs/benchmark/two_stage_tsvd_multistage.jl
#
# Stage s fits a fresh `ScaleThenReflect` in the frame of stages 1..s-1 (stage 1
# is the shipped fit, stage 2 is `TwoStageScaleThenReflect`'s second pass). The
# composite is L = S₁ S₂ ⋯; fitting stops when a stage adds no reflection or at
# WHMC_TSVD_MAX_STAGES. Per stage: reflections added, coordinates still bad,
# kappa of the composite so far, and the stage's fit time.

using LinearAlgebra
BLAS.set_num_threads(1)
using Printf
include("two_stage_tsvd_common.jl")

const N_WINDOW = parse(Int, get(ENV, "WHMC_TSVD_N", "1000"))
const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "2"))
const DIMS = parse.(Int, split(get(ENV, "WHMC_TSVD_DIMS", "100,200,400"), ","))
const MAX_STAGES = parse(Int, get(ENV, "WHMC_TSVD_MAX_STAGES", "6"))
const CASES = split(get(ENV, "WHMC_TSVD_CASES", "pairs:spread,ar1:spread,dense:spread,hub:spread"), ",")

for d in DIMS, case in CASES, seed in 1:N_SEEDS
    st, sc = split(case, ":")
    Σ = gaussian_covariance(st, sc, d, seed)
    P = Symmetric(inv(Σ))
    rng = Xoshiro(hash((d, st, sc, seed, :window)))
    x = cholesky(Σ).L * randn(rng, d, N_WINDOW)
    g = -(P * x)
    L = nothing
    z, gz = x, g
    for s in 1:MAX_STAGES
        t = @elapsed ((S, loss) = fit(W._scale_then_reflect, z, gz; UNBOUNDED...))
        added = W.n_reflections(S)
        L = isnothing(L) ? S : W.MatrixFactorization(L, S)
        z, gz = S \ z, S' * gz
        @printf("%-4d %-6s %-7s seed=%d stage=%d added=%-4d bad=%-4d loss=%.3f kappa=%.4g fit_s=%.2f\n",
                d, st, sc, seed, s, added, bad_left(L, x, g), loss, kappa(L, P), t)
        flush(stdout)
        added == 0 && break
    end
end
