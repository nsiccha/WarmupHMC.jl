# What does capping the number of successive reflections cost in fit quality,
# and what does it save? (WarmupHMC go/no-go `WarmupHMC:two-stage-tsvd/decisions/
# 2026-10-22T12-42-03-480-04q64y5`: keep the metric linear in d; the user asked
# for a bound with a smallish, tunable default.)
#
#   julia --project=. docs/benchmark/reflection_cap_fit.jl
#
# Same exact-Gaussian windows and measures as `two_stage_tsvd_fit.jl`
# (`two_stage_tsvd_common.jl`), fitting the shipped `ScaleThenReflect` with
# `update_loss!(…; max_reflections=cap)` for each cap in WHMC_CAPS (`0` = no cap).
# Reported per fit: selection loss, kappa (true condition number), kappa_new
# (fit to a fresh window), reflections, coordinates left bad, fit ms, and the
# metric mat-vec in µs.

using LinearAlgebra
BLAS.set_num_threads(1)
using Printf
include("two_stage_tsvd_common.jl")

const N_WINDOW = parse(Int, get(ENV, "WHMC_TSVD_N", "1000"))
const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "3"))
const DIMS = parse.(Int, split(get(ENV, "WHMC_TSVD_DIMS", "20,100,400"), ","))
const CAPS = parse.(Int, split(get(ENV, "WHMC_CAPS", "0,32,16,8,4,2"), ","))
const OUT = get(ENV, "WHMC_TSVD_OUT", "")

window(rng, Lchol, P, n) = (x = Lchol * randn(rng, size(P, 1), n); (x, -(P * x)))
cap_kw(cap) = cap == 0 ? UNBOUNDED : (; max_reflections=cap)

const ROW_KEYS = (:d, :structure, :scaling, :seed, :cap, :loss, :kappa, :kappa_new,
                  :kappa_diag, :refl, :bad_left, :fit_ms, :matvec_us)
isempty(OUT) || open(io -> println(io, join(ROW_KEYS, '\t')), OUT, "w")
rows = NamedTuple[]
for d in DIMS, st in STRUCTURES, sc in SCALINGS, seed in 1:N_SEEDS
    rng = Xoshiro(hash((d, st.name, sc.name, seed, :window)))
    Σ = gaussian_covariance(st.name, sc.name, d, seed)
    P = Symmetric(inv(Σ))
    Lchol = cholesky(Σ).L
    x, g = window(rng, Lchol, P, N_WINDOW)
    xf, gf = window(rng, Lchol, P, N_WINDOW)
    kappa_diag = kappa(Diagonal(sqrt.(diag(Σ))), P)
    for cap in CAPS
        t = @elapsed ((L, loss) = fit(W._scale_then_reflect, x, g; cap_kw(cap)...))
        Lf, _ = fit(W._scale_then_reflect, xf, gf; cap_kw(cap)...)
        fit_ms = d >= 400 ? 1e3t : timed_fit(W._scale_then_reflect, x, g; reps=3, cap_kw(cap)...)
        row = NamedTuple{ROW_KEYS}((d, st.name, sc.name, seed, cap, loss, kappa(L, P), kappa(Lf, P),
                                    kappa_diag, W.n_reflections(L), bad_left(L, x, g), fit_ms, matvec_us(L)))
        push!(rows, row)
        isempty(OUT) || open(io -> println(io, join(values(row), '\t')), OUT, "a")
        @printf("# %-4d %-7s %-7s seed=%d cap=%-3d loss=%.3f kappa=%.3g kappa_new=%.3g refl=%d bad=%d fit_ms=%.1f matvec_us=%.2f\n",
                d, st.name, sc.name, seed, cap, row.loss, row.kappa, row.kappa_new, row.refl,
                row.bad_left, row.fit_ms, row.matvec_us)
        flush(stdout)
    end
end

println(rpad("d", 5), rpad("structure", 9), rpad("scaling", 8), lpad("cap", 4),
        lpad("loss", 8), lpad("kappa", 11), lpad("kappa_new", 11), lpad("refl", 6),
        lpad("bad", 5), lpad("fit_ms", 9), lpad("mv_us", 8), lpad("k_diag", 11))
for d in DIMS, st in STRUCTURES, sc in SCALINGS, cap in CAPS
    r = filter(r -> r.d == d && r.structure == st.name && r.scaling == sc.name && r.cap == cap, rows)
    m(f) = median(map(f, r))
    @printf("%-5d%-9s%-8s%4d%8.3f%11.3g%11.3g%6.0f%5.0f%9.2f%8.2f%11.3g\n",
            d, st.name, sc.name, cap, m(r -> r.loss), m(r -> r.kappa), m(r -> r.kappa_new),
            m(r -> r.refl), m(r -> r.bad_left), m(r -> r.fit_ms), m(r -> r.matvec_us), m(r -> r.kappa_diag))
end
