# Model-specific reparametrization factories for PosteriorDB posteriors.
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

gp_log_scale(sdgp, lscale, i; slambda) = begin
    @. .5 * (2*sdgp + .5 * log(2pi) + lscale -.5 * exp(2*lscale) * abs2(slambda[i]))
end

"""
    _index_getter(i) -> (x -> x[i])

Build a coordinate accessor **in its own function scope**.

DO NOT inline this back into the `elseif` chain below, and do not write
`x -> x[l]` there directly. `l`, `s` and `o` are assigned in several branches of
that one long `if/elseif` scope, so Julia's closure conversion cannot prove
single assignment and captures a `Core.Box` — a mutable heap cell read as `Any`
— instead of an `Int`. As a function parameter, `i` is single-assignment and is
captured by value.

These closures run under AD on the gradient hot path (see the
`ReparametrizedProblem` docstring), so the boxing was not a micro-detail. On
`radon_partially_pooled`, same 85 pairs and same indices, gradients agreeing to
exactly `0.0`:

| spec | ForwardDiff | Enzyme/`Const` |
|---|---|---|
| boxed    | 339590 ns | 429477 ns |
| de-boxed |  79567 ns |  27911 ns |

It also **inverted the apparent backend ranking** — Enzyme measured 1.25–1.52×
slower than ForwardDiff on that target while boxed, and 2.85× faster de-boxed —
and since the boxed specs happened to be the larger models, it manufactured a
convincing "reverse mode loses at high dimension" pattern out of nothing. Two
docstrings published that artifact as a property of the AD backend before this
was found. Found by `reparam-bench` auditing its own benchmark.
"""
_index_getter(i::Integer) = x -> x[i]

# One `Reparametrization` per group coordinate `o+1 … o+n`, sharing a location
# and a log-scale. Every index is a parameter here, so nothing boxes.
_grouped_pairs(c, n, o, location, log_scale) = begin
    r = Reparametrization(PartiallyCentered(c), PartiallyCentered(c),
                          location, log_scale)
    [(o + i) => r for i in 1:n]
end

reparametrization(posterior_name, dim, stan_jdata) = if startswith(posterior_name, "funnel")
    c = 1.
    2:dim .=> Ref(Reparametrization(
        PartiallyCentered(c),
        PartiallyCentered(c),
        0.,
        x->x[1]
    ))

elseif !isnothing(match(r"-eight_schools_(non|)centered", posterior_name))
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    1:8 .=> Ref(Reparametrization(
        PartiallyCentered(c),
        PartiallyCentered(c),
        x->x[9],
        x->x[10]
    ))
elseif !isnothing(match(r"-seeds_(non|)centered", posterior_name))
    I = stan_jdata["I"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    _grouped_pairs(c, I, 4, 0., _index_getter(I+5))
elseif !isnothing(match(r"-radon_partially_pooled_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    _grouped_pairs(c, J, 0, _index_getter(J+1), _index_getter(J+2))
elseif !isnothing(match(r"-radon_variable_intercept_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    _grouped_pairs(c, J, 0, _index_getter(J+2), _index_getter(J+3))
elseif !isnothing(match(r"-radon_variable_slope_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    _grouped_pairs(c, J, 1, _index_getter(J+2), _index_getter(J+3))
elseif !isnothing(match(r"-radon_hierarchical_intercept_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    _grouped_pairs(c, J, 0, _index_getter(J+3), _index_getter(J+4))
elseif !isnothing(match(r"-radon_variable_intercept_slope_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    mapreduce(vcat, ((4+2*J, 2, 3), (5+2*J, 3, 3+J))) do (l, s, o)
        _grouped_pairs(c, J, o, _index_getter(l), _index_getter(s))
    end
elseif !isnothing(match(r"-accel_gp", posterior_name))
    mapreduce(vcat, [(40, 1, "slambda_1"), (20, 44, "slambda_sigma_1")]) do (n, o, slambda_name)
        slambda = stan_jdata[slambda_name]
        map(1:n) do i
            idx = o+2+i
            idx=>Reparametrization(
                PartiallyCentered(0.),
                PartiallyCentered(0.),
                0.,
                x->gp_log_scale(x[o+1], x[o+2], i; slambda)
            )
        end
    end
else
    Nothing[]
end |> IndexedReparametrization
