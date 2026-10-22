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
#   fit_ms     median wall time of one `update_loss!` call (BLAS single-threaded).

using LinearAlgebra
BLAS.set_num_threads(1)
using WarmupHMC, Random, Statistics, Printf

const W = WarmupHMC
const N_WINDOW = parse(Int, get(ENV, "WHMC_TSVD_N", "1000"))
const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "5"))
const DIMS = parse.(Int, split(get(ENV, "WHMC_TSVD_DIMS", "20,100,400"), ","))
const OUT = get(ENV, "WHMC_TSVD_OUT", "")

const VARIANTS = [
    (name="current", make=W._scale_then_reflect),
    (name="scaled", make=W._scaled_scale_then_reflect),
    (name="two_stage", make=W._two_stage_scale_then_reflect),
]

# ── correlation structures (unit marginal variance) ──────────────────────────
to_correlation(S) = (s = sqrt.(diag(S)); Symmetric(S ./ (s * s')))

# Independent strongly correlated pairs (ρ ∈ [0.9, 0.995]): exactly separable.
corr_pairs(rng, d) = begin
    C = Matrix(1.0I, d, d)
    for i in 1:2:d-1
        C[i, i+1] = C[i+1, i] = 0.9 + 0.095rand(rng)
    end
    Symmetric(C)
end
# One shared latent factor for the first half (a hierarchical "global" effect),
# independent coordinates for the second half.
corr_factor(rng, d) = begin
    w = zeros(d); w[1:d÷2] .= 1 .+ 3rand(rng, d ÷ 2)
    to_correlation(w * w' + I)
end
# Dense: k = d/4 random factors.
corr_dense(rng, d) = (Wf = randn(rng, d, max(1, d ÷ 4)); to_correlation(Wf * Wf' + 0.05I))
# AR(1) chain, ρ = 0.95 (a random walk / time-series prior).
corr_ar1(rng, d) = Symmetric([0.95^abs(i - j) for i in 1:d, j in 1:d])
# Pairs plus a hub: coordinate 1 is weakly tied to many otherwise independent
# coordinates (each tie too weak to make them bad, the sum enough to make the
# hub bad). Built on the precision so the ties are direct.
corr_hub(rng, d) = begin
    m = d ÷ 2
    Pm = Matrix(1.0I, d, d)
    for j in 2:m
        Pm[1, j] = Pm[j, 1] = -0.9 / sqrt(m - 1)
    end
    C = to_correlation(inv(Symmetric(Pm)))
    Cm = Matrix(C)
    for i in m+1:2:d-1
        Cm[i, i+1] = Cm[i+1, i] = 0.97
    end
    Symmetric(Cm)
end

const STRUCTURES = [
    (name="pairs", f=corr_pairs),
    (name="factor", f=corr_factor),
    (name="dense", f=corr_dense),
    (name="ar1", f=corr_ar1),
    (name="hub", f=corr_hub),
]
const SCALINGS = [
    (name="equal", f=(rng, d) -> ones(d)),
    # Log-uniform over six orders of magnitude, randomly assigned.
    (name="spread", f=(rng, d) -> exp10.(-3 .+ 6rand(rng, d))),
]

window(rng, Lchol, P, n) = (x = Lchol * randn(rng, size(P, 1), n); (x, -(P * x)))

dense(L) = L * Matrix{Float64}(I, size(L)...)
kappa(L, P) = begin
    Lm = dense(L)
    λ = eigvals(Symmetric(Lm' * P * Lm))
    λ[end] / λ[1]
end
bad_left(L, x, g; threshold=log(2)) = begin
    z, gz = L \ x, L' * g
    count(i -> abs2(log(std(view(z, i, :)) * std(view(gz, i, :)))) >= threshold, axes(z, 1))
end

function fit(make, x, g)
    L = make(size(x, 1))
    loss = W.update_loss!(L, x, g)
    L, loss
end

# Always on copies: the shipped `update_loss!(::ScaleThenReflect)` reflects its
# inputs in place.
function timed_fit(make, x, g; reps=5)
    fit(make, copy(x), copy(g))
    ts = map(1:reps) do _
        xc, gc = copy(x), copy(g)
        @elapsed fit(make, xc, gc)
    end
    1e3median(ts)
end

rows = NamedTuple[]
for d in DIMS, st in STRUCTURES, sc in SCALINGS, seed in 1:N_SEEDS
    rng = Xoshiro(hash((d, st.name, sc.name, seed)))
    C = st.f(rng, d)
    s = sc.f(rng, d)
    Σ = Symmetric(Diagonal(s) * C * Diagonal(s))
    P = Symmetric(inv(Σ))
    Lchol = cholesky(Σ).L
    x, g = window(rng, Lchol, P, N_WINDOW)
    xf, gf = window(rng, Lchol, P, N_WINDOW)
    kappa_target = (λ = eigvals(Σ); λ[end] / λ[1])
    kappa_diag = kappa(Diagonal(sqrt.(diag(Σ))), P)
    for v in VARIANTS
        L, loss = fit(v.make, copy(x), copy(g))
        Lf, _ = fit(v.make, copy(xf), copy(gf))
        row = (;
            d, structure=st.name, scaling=sc.name, seed, variant=v.name,
            loss, kappa=kappa(L, P), kappa_new=kappa(Lf, P),
            kappa_target, kappa_diag,
            refl=W.n_reflections(L), bad_left=bad_left(L, x, g),
            fit_ms=timed_fit(v.make, x, g),
        )
        push!(rows, row)
    end
end

isempty(OUT) || open(OUT, "w") do io
    println(io, join(keys(first(rows)), '\t'))
    foreach(r -> println(io, join(values(r), '\t')), rows)
end

# ── summary: medians over seeds ──────────────────────────────────────────────
using Statistics: median
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
