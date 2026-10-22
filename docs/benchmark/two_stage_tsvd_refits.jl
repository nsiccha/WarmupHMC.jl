# Every metric refit of real `adaptive_warmup_mcmc` runs, diagnosed: why the
# shipped reflection fit stopped, and whether a second pass after the diagonal
# (or any of the next singular directions) would have found more.
#
#   WHMC_TSVD_OUT=refits.jls julia --project=docs/benchmark docs/benchmark/two_stage_tsvd_refits.jl
#
# Answers the user's feedback on decision
# `WarmupHMC:two-stage-tsvd/decisions/2026-10-22T12-41-51-077-01buaqu` (#1): judge it on
# real, non-Gaussian posteriors, at the windows where the sampler actually refits.
#
# The sampler runs unchanged (shipped metric). A narrower method of the internal
# `WarmupHMC._select_transformation!`, defined in THIS process only, records the
# diagnostics on copies of the halo the window refits on and then `invoke`s the
# shipped method, so selection and sampling are exactly the shipped ones.
#
# Per refit:
#   d, n            dimension and halo size
#   bad0            coordinates the diagonal fits badly (log(s1·s2)² ≥ log 2)
#   refl, bad       reflections the shipped fit added (unbounded) and coordinates left bad
#   stop            :none_left | :direction_failed
#   l_raw, l_scaled 1-D loss of the stopping direction (threshold log 2 = 0.693):
#                   the top gradient singular vector of the bad block in the
#                   reflected frame, and in that frame after the diagonal
#   l5_raw, l5_scaled  the best 1-D loss among the top 5 singular vectors there
#   refl2           reflections a second pass after the diagonal adds
#   loss, loss2     selection loss of the shipped fit and of the two-stage fit
#   fit_ms, fit2_ms one timed shipped fit and one two-stage fit on this halo

using LinearAlgebra
BLAS.set_num_threads(1)
include("common.jl")
include("two_stage_tsvd_common.jl")
using Serialization

const N_SEEDS = parse(Int, get(ENV, "WHMC_TSVD_SEEDS", "4"))
const N_DRAWS = parse(Int, get(ENV, "WHMC_TSVD_DRAWS", "1000"))
const OUT = ENV["WHMC_TSVD_OUT"]
const THRESHOLD = log(2)
const ALL_TARGETS = [
    "kilpisjarvi_mod-kilpisjarvi", "earnings-logearn_interaction", "garch-garch11",
    "low_dim_gauss_mix-low_dim_gauss_mix", "arK-arK", "eight_schools-eight_schools_centered",
    "nes2000-nes", "gp_pois_regr-gp_pois_regr", "surgical_data-surgical_model",
    "pilots-pilots", "diamonds-diamonds", "seeds_data-seeds_centered_model",
    "mcycle_gp-accel_gp", "radon_mn-radon_partially_pooled_centered", "irt_2pl-irt_2pl",
    "funnel",
]
const SELECTED = let raw = get(ENV, "WHMC_TSVD_TARGETS", "")
    isempty(raw) ? ALL_TARGETS : String.(strip.(split(raw, ",")))
end

coordinate_losses(p, g) = abs2.(log.(std.(eachrow(p)) .* std.(eachrow(g))))
direction_loss(v, p, g) = abs2(log(std(v' * p) * std(v' * g)))
# 1-D losses of the top `k` left singular vectors of the bad block's gradients.
top_direction_losses(p, g, k) = begin
    size(g, 1) == 1 && return [direction_loss(ones(1), p, g)]
    U = WarmupHMC.tsvd(g, min(k, size(g, 1) - 1); initvec=ones(size(g, 1)))[1]
    [direction_loss(u, p, g) for u in eachcol(U)]
end

# R⁻¹ applied column by column (R is orthogonal, so this is also R' for gradients).
unreflect(R, x) = reduce(hcat, [ldiv!(similar(c), R, c) for c in eachcol(x)])

const REFITS = NamedTuple[]
const CONTEXT = Ref((; target="", seed=0))

function diagnose(x, g)
    d, n = size(x)
    bad0 = count(>=(THRESHOLD), coordinate_losses(x, g))
    t = @elapsed ((L, loss) = fit(W._scale_then_reflect, x, g; UNBOUNDED...))
    t2 = @elapsed ((L2, loss2) = fit(W._two_stage_scale_then_reflect, x, g; UNBOUNDED...))
    R, D = L.m1, L.m2
    z, gz = unreflect(R, x), unreflect(R, g)  # the reflected frame the loop stopped in
    zs, gzs = D \ z, D' * gz                  # the same frame after the diagonal
    bad = findall(>=(THRESHOLD), coordinate_losses(z, gz))
    l_raw = l_scaled = l5_raw = l5_scaled = NaN
    if !isempty(bad)
        lr = top_direction_losses(z[bad, :], gz[bad, :], 5)
        ls = top_direction_losses(zs[bad, :], gzs[bad, :], 5)
        l_raw, l5_raw, l_scaled, l5_scaled = lr[1], minimum(lr), ls[1], minimum(ls)
    end
    push!(REFITS, (; CONTEXT[]..., d, n, bad0, refl=W.n_reflections(L), bad=length(bad),
                     stop=isempty(bad) ? :none_left : :direction_failed,
                     l_raw, l_scaled, l5_raw, l5_scaled,
                     refl2=W.n_reflections(L2.m2), loss, loss2,
                     fit_ms=1e3t, fit2_ms=1e3t2))
end

# Narrower than the shipped `_select_transformation!(scale_options, active,
# positions, gradients)`: diagnose, then run the shipped method itself.
function WarmupHMC._select_transformation!(scale_options::NamedTuple, active::Symbol,
                                           positions::AbstractMatrix, gradients::AbstractMatrix; kwargs...)
    size(positions, 2) > 1 && diagnose(Matrix{Float64}(positions), Matrix{Float64}(gradients))
    invoke(WarmupHMC._select_transformation!, Tuple{Any,Any,Any,Any},
           scale_options, active, positions, gradients; kwargs...)
end

for name in SELECTED
    problem = name == "funnel" ? Funnel(10) : first(stan_problem(name))
    for seed in 1:N_SEEDS
        CONTEXT[] = (; target=name, seed)
        before = length(REFITS)
        try
            adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws=N_DRAWS,
                                 nonlinear_adapt=false, monitor_ess=false, progress=nothing)
        catch err
            push!(REFITS, (; CONTEXT[]..., error=sprint(showerror, err)))
        end
        serialize(OUT, REFITS)
        new = REFITS[before+1:end]
        println(rpad(name, 45), " seed=", seed, " refits=", length(new),
                " refl2>0 in ", count(r -> get(r, :refl2, 0) > 0, new))
        flush(stdout)
    end
end
