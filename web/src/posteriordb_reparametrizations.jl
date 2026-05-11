# Model-specific reparametrization factories for PosteriorDB posteriors.
# Migrated from LocalScalesHMC.jl/julia/reparametrizations.jl

gp_log_scale(sdgp, lscale, i; slambda) = begin
    @. .5 * (2*sdgp + .5 * log(2pi) + lscale -.5 * exp(2*lscale) * abs2(slambda[i]))
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
    (s, o) = (I+5, 4)
    map(1:I) do i
        idx = o + i
        idx=>Reparametrization(
            PartiallyCentered(c),
            PartiallyCentered(c),
            0.,
            x->x[s]
        )
    end
elseif !isnothing(match(r"-radon_partially_pooled_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    (l, s, o) = (J+1, J+2, 0)
    map(1:J) do i
        idx = o + i
        idx=>Reparametrization(
            PartiallyCentered(c),
            PartiallyCentered(c),
            x->x[l],
            x->x[s]
        )
    end
elseif !isnothing(match(r"-radon_variable_intercept_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    (l, s, o) = (J+2, J+3, 0)
    map(1:J) do i
        idx = o + i
        idx=>Reparametrization(
            PartiallyCentered(c),
            PartiallyCentered(c),
            x->x[l],
            x->x[s]
        )
    end
elseif !isnothing(match(r"-radon_variable_slope_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    (l, s, o) = (J+2, J+3, 1)
    map(1:J) do i
        idx = o + i
        idx=>Reparametrization(
            PartiallyCentered(c),
            PartiallyCentered(c),
            x->x[l],
            x->x[s]
        )
    end
elseif !isnothing(match(r"-radon_hierarchical_intercept_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    (l, s, o) = (J+3, J+4, 0)
    map(1:J) do i
        idx = o + i
        idx=>Reparametrization(
            PartiallyCentered(c),
            PartiallyCentered(c),
            x->x[l],
            x->x[s]
        )
    end
elseif !isnothing(match(r"-radon_variable_intercept_slope_(non|)centered", posterior_name))
    J = stan_jdata["J"]
    c = endswith(posterior_name, "noncentered") ? 0. : 1.
    mapreduce(vcat, ((4+2*J, 2, 3), (5+2*J, 3, 3+J))) do (l, s, o)
        map(1:J) do i
            idx = o + i
            idx=>Reparametrization(
                PartiallyCentered(c),
                PartiallyCentered(c),
                x->x[l],
                x->x[s]
            )
        end
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
