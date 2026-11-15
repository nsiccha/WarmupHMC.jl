# Shared by the `reflection_search_*.jl` drivers: the `:adaptive` reflection-fit
# variants under comparison and the measures of a fitted frame.
#
# Decision `WarmupHMC:two-stage-tsvd/decisions/2026-10-22T12-41-51-077-01buaqu`
# (user pick: search further directions before stopping, in the raw and the
# rescaled frame, TSVD on a contiguous copy); todo
# `WarmupHMC:two-stage-tsvd/todos/2026-10-22T12-43-40-337-1l56lb2`.
#
# Every variant keeps the shipped `max_reflections` bound (16): that is what a
# landing would ship. The variants differ only in the keywords the experimental
# `update_loss!(::SuccessiveReflections)` takes and in the structure of the
# candidate (`R D` or the two-stage `R₁ D₁ R₂ D₂`).

using LinearAlgebra, Statistics
using WarmupHMC
const W = WarmupHMC

const VARIANTS = [
    # The rule on `dev`: stop at the first direction that fails the 1-D check.
    (name="shipped", make=W._scale_then_reflect, kw=(;)),
    # The same rule, TSVD on a contiguous copy of the bad block.
    (name="contig", make=W._scale_then_reflect, kw=(; contiguous=true)),
    # Raw (reflected) frame: when the top direction fails, take the first of the
    # leading 5 that passes.
    (name="topk", make=W._scale_then_reflect, kw=(; contiguous=true, n_directions=5)),
    # ... the one of the leading 5 with the smallest passing loss.
    (name="topk_best", make=W._scale_then_reflect, kw=(; contiguous=true, n_directions=5, pick=:best)),
    # Sensitivity to k: the leading 10.
    (name="topk10", make=W._scale_then_reflect, kw=(; contiguous=true, n_directions=10)),
    # The shipped rule, then a second reflection loop after the diagonal.
    (name="two_stage", make=W._two_stage_scale_then_reflect, kw=(; contiguous=true)),
    # The top-5 search in the raw frame, then again in the rescaled frame.
    (name="topk_two_stage", make=W._two_stage_scale_then_reflect, kw=(; contiguous=true, n_directions=5)),
]
variant(name) = only(filter(v -> v.name == name, VARIANTS))

const THRESHOLD = log(2)

# One fit of a fresh candidate to COPIES of a window (`update_loss!` reflects,
# and the two-stage fit also rescales, its inputs in place); returns (L, loss).
function fit(v, x, g)
    L = v.make(size(x, 1))
    loss = W.update_loss!(L, copy(x), copy(g); v.kw...)
    L, loss
end

# Fastest of `reps` timed fits (ms), after the one the caller already ran.
fit_ms(v, x, g; reps=3) = 1e3minimum(@elapsed(fit(v, x, g)) for _ in 1:reps)

n_reflections(L) = W.n_reflections(L)

# Per-coordinate 1-D losses of frame `L` on (x, g): log(std(z_i)·std(gz_i))².
coordinate_losses(L, x, g) = begin
    z, gz = L \ x, L' * g
    [abs2(log(std(view(z, i, :)) * std(view(gz, i, :)))) for i in axes(z, 1)]
end
bad_left(L, x, g) = count(>=(THRESHOLD), coordinate_losses(L, x, g))

# The selection loss of frame `L` on (x, g) as it stands, refitting nothing (the
# quantity the restart test compares): mean over coordinates of std·std.
frame_loss(L, x, g) = W._frame_loss(L, x, g)

const POSTERIORDB_TARGETS = [
    "kilpisjarvi_mod-kilpisjarvi", "earnings-logearn_interaction", "garch-garch11",
    "low_dim_gauss_mix-low_dim_gauss_mix", "arK-arK", "eight_schools-eight_schools_centered",
    "nes2000-nes", "gp_pois_regr-gp_pois_regr", "surgical_data-surgical_model",
    "pilots-pilots", "diamonds-diamonds", "seeds_data-seeds_centered_model",
    "mcycle_gp-accel_gp", "radon_mn-radon_partially_pooled_centered", "irt_2pl-irt_2pl",
]
const ALL_TARGETS = [POSTERIORDB_TARGETS; "funnel"]
selected_targets() = let raw = get(ENV, "WHMC_RS_TARGETS", "")
    isempty(raw) ? ALL_TARGETS : String.(strip.(split(raw, ",")))
end
make_problem(name) = name == "funnel" ? Funnel(10) : first(stan_problem(name))
