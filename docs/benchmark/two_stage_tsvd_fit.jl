# Offline comparison of the `:adaptive` metric fit variants on Gaussian windows.
#
#   julia --project=. docs/benchmark/two_stage_tsvd_fit.jl
#
# Needs only WarmupHMC and stdlibs, so it runs in the package's own environment.
# `WHMC_TSVD_OUT=<file>` also writes every row as tab-separated values.
#
# Investigation todo `WarmupHMC:two-stage-tsvd/todos/2026-10-22T10-55-12-401-1ruyk25`
# (the user's question: does a second TSVD after the diagonal scaling step find
# better reflection directions than the single TSVD on the raw gradients?).
#
# Every variant is fitted to the SAME window: `n` exact draws `x ~ N(0, Σ)` and
# their gradients `g = -Σ⁻¹ x`, the evidence `update_loss!` sees each window.
#
#   current   L = R D           (`ScaleThenReflect`, the shipped fit)
#   scaled    L = D₀ R D        (`ScaledScaleThenReflect`: diagonal first, one TSVD pass in its frame)
#   two_stage L = R₁ D₁ R₂ D₂   (`TwoStageScaleThenReflect`: the shipped fit, then a second pass)
#
# Reported per fit:
#   loss       what `update_loss!` returns (mean s1·s2 in the fitted frame; 1 = perfect).
#              This is the in-sample number the sampler selects metrics by.
#   kappa      condition number of the preconditioned precision L' Σ⁻¹ L: the
#              truth the loss estimates. NUTS's leapfrog steps per independent
#              draw grow roughly like sqrt(kappa). Dense d × d eigenvalues —
#              evaluation only, never on a sampler path.
#   kappa_new  the same for a fit to a FRESH window from the same target, so the
#              number is not flattered by in-sample overfit of a single window.
#   refl       reflections stored (stage 1 + stage 2 for two_stage).
#   bad_left   coordinates still at or above the reflection threshold afterwards.
#   fit_ms     wall time of one `update_loss!` call (BLAS single-threaded):
#              median of WHMC_TSVD_REPS calls, a single call at d ≥ 400.

using LinearAlgebra
BLAS.set_num_threads(1)
using Printf
include("two_stage_tsvd_common.jl")

const N_WINDOW = parse(Int, get(ENV, "WHMC_TSVD_N", "1000"))
const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "5"))
const DIMS = parse.(Int, split(get(ENV, "WHMC_TSVD_DIMS", "20,100,400"), ","))
const OUT = get(ENV, "WHMC_TSVD_OUT", "")
const REPS = parse(Int, get(ENV, "WHMC_TSVD_REPS", "3"))

window(rng, Lchol, P, n) = (x = Lchol * randn(rng, size(P, 1), n); (x, -(P * x)))

const ROW_KEYS = (:d, :structure, :scaling, :seed, :variant, :loss, :kappa, :kappa_new,
                  :kappa_target, :kappa_diag, :refl, :bad_left, :fit_ms)
isempty(OUT) || open(io -> println(io, join(ROW_KEYS, '\t')), OUT, "w")

rows = NamedTuple[]
for d in DIMS, st in STRUCTURES, sc in SCALINGS, seed in 1:N_SEEDS
    rng = Xoshiro(hash((d, st.name, sc.name, seed, :window)))
    Σ = gaussian_covariance(st.name, sc.name, d, seed)
    P = Symmetric(inv(Σ))
    Lchol = cholesky(Σ).L
    x, g = window(rng, Lchol, P, N_WINDOW)
    xf, gf = window(rng, Lchol, P, N_WINDOW)
    kappa_target = (λ = eigvals(Σ); λ[end] / λ[1])
    kappa_diag = kappa(Diagonal(sqrt.(diag(Σ))), P)
    for v in VARIANTS
        t = @elapsed ((L, loss) = fit(v.make, x, g; UNBOUNDED...))
        Lf, _ = fit(v.make, xf, gf; UNBOUNDED...)
        # One timed call at large d (a fit there takes seconds); median of
        # REPS after a warm-up otherwise.
        fit_ms = d >= 400 ? 1e3t : timed_fit(v.make, x, g; reps=REPS, UNBOUNDED...)
        row = NamedTuple{ROW_KEYS}((
            d, st.name, sc.name, seed, v.name, loss, kappa(L, P), kappa(Lf, P),
            kappa_target, kappa_diag, W.n_reflections(L), bad_left(L, x, g), fit_ms,
        ))
        push!(rows, row)
        isempty(OUT) || open(io -> println(io, join(values(row), '\t')), OUT, "a")
        @printf("# %-4d %-7s %-7s seed=%d %-10s loss=%.3f kappa=%.3g kappa_new=%.3g refl=%d bad=%d fit_ms=%.1f\n",
                d, st.name, sc.name, seed, v.name, row.loss, row.kappa, row.kappa_new,
                row.refl, row.bad_left, row.fit_ms)
        flush(stdout)
    end
end

# ── summary: medians over seeds ──────────────────────────────────────────────
println(rpad("d", 5), rpad("structure", 9), rpad("scaling", 8), rpad("variant", 10),
        lpad("loss", 8), lpad("kappa", 11), lpad("kappa_new", 11), lpad("refl", 6),
        lpad("bad", 5), lpad("fit_ms", 9), lpad("k_diag", 11))
for d in DIMS, st in STRUCTURES, sc in SCALINGS, v in VARIANTS
    r = filter(r -> r.d == d && r.structure == st.name && r.scaling == sc.name && r.variant == v.name, rows)
    m(f) = median(map(f, r))
    @printf("%-5d%-9s%-8s%-10s%8.3f%11.3g%11.3g%6.0f%5.0f%9.2f%11.3g\n",
            d, st.name, sc.name, v.name, m(r -> r.loss), m(r -> r.kappa), m(r -> r.kappa_new),
            m(r -> r.refl), m(r -> r.bad_left), m(r -> r.fit_ms), m(r -> r.kappa_diag))
end
