# Shared by `two_stage_tsvd_fit.jl` and `two_stage_tsvd_sample.jl`: the metric
# fit variants under comparison, the synthetic Gaussian targets, and the
# metric-quality measures. Investigation todo
# `WarmupHMC:two-stage-tsvd/todos/2026-10-22T10-55-12-401-1ruyk25`.

using LinearAlgebra, Random, Statistics
using WarmupHMC
const W = WarmupHMC

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

"""
    gaussian_covariance(structure, scaling, d, seed) -> Σ

The synthetic target `structure` × `scaling` at dimension `d`, deterministic in
its arguments.
"""
gaussian_covariance(structure::AbstractString, scaling::AbstractString, d, seed) = begin
    rng = Xoshiro(hash((d, structure, scaling, seed)))
    st = only(filter(s -> s.name == structure, STRUCTURES))
    sc = only(filter(s -> s.name == scaling, SCALINGS))
    C = st.f(rng, d)
    s = sc.f(rng, d)
    Symmetric(Diagonal(s) * C * Diagonal(s))
end

dense(L) = L * Matrix{Float64}(I, size(L)...)
kappa(L, P) = begin
    Lm = dense(L)
    λ = eigvals(Symmetric(Lm' * P * Lm))
    λ[end] / λ[1]
end
# The selection loss of frame `L` on positions `x` and gradients `g`, the
# quantity `update_loss!` returns (mean over coordinates of s1·s2, 1 = fitted),
# evaluated on evidence the fit did not see when `x`, `g` are fresh draws.
frame_loss(L, x, g) = begin
    z, gz = L \ x, L' * g
    mean(i -> std(view(z, i, :)) * std(view(gz, i, :)), axes(z, 1))
end
bad_left(L, x, g; threshold=log(2)) = begin
    z, gz = L \ x, L' * g
    count(i -> abs2(log(std(view(z, i, :)) * std(view(gz, i, :)))) >= threshold, axes(z, 1))
end

# One fit of a fresh `make(d)` to (copies of) a window; returns (L, loss).
# Always on copies: the shipped `update_loss!(::ScaleThenReflect)` reflects its
# inputs in place.
# The two-TSVD investigation measured the fit without a reflection bound (it
# predates `max_reflections`); its drivers pass this to reproduce those numbers.
const UNBOUNDED = (; max_reflections=typemax(Int))

# `kwargs` go to `update_loss!` (e.g. `max_reflections`).
function fit(make, x, g; kwargs...)
    L = make(size(x, 1))
    loss = W.update_loss!(L, copy(x), copy(g); kwargs...)
    L, loss
end

# Median wall time (ms) of one fit, after a warm-up fit.
function timed_fit(make, x, g; reps=5, kwargs...)
    fit(make, x, g; kwargs...)
    1e3median(map(_ -> (@elapsed fit(make, x, g; kwargs...)), 1:reps))
end

# Median wall time (µs) of one metric mat-vec `mul!(y, L, z)`.
matvec_us(L) = begin
    z = randn(Xoshiro(0), size(L, 1)); y = similar(z)
    mul!(y, L, z)
    1e6minimum(@elapsed(for _ in 1:100; mul!(y, L, z); end) for _ in 1:3) / 100
end
