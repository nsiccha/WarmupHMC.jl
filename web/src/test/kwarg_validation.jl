# Tests for entry-point keyword validation (src/kwarg_validation.jl):
#   * the DRIFT GUARD — `_SAMPLER_KWARGS` must still match the live method
#     signatures, so adding a kwarg without updating the table fails here
#     instead of silently narrowing the accepted surface;
#   * the four silent-failure modes this exists to close (unhonourable kwarg,
#     typo, foreign idiom, wrong-method kwarg) now throw;
#   * legitimate calls, including deliberate Pathfinder passthrough via
#     `pathfinder_kw`, still work.
#
# Run this item alone with
#   -- --file=kwarg_validation.jl

@testitem "entry-point keyword validation" setup=[Determinism] tags=[:sampler] begin
    using WarmupHMC, LogDensityProblems, LinearAlgebra, Random
    using WarmupHMC: _SAMPLER_KWARGS, _INITIALIZER_KWARGS, _MULTICHAIN_ONLY_KWARGS,
        _check_kwargs, _edit_distance, _nearest_kwarg, clustered_chains

    # --- drift guard --------------------------------------------------------------
    # The accepted sets are literal tuples (so include order does not matter and
    # there is no per-call reflection). That is only safe if something checks them
    # against the real signatures — this is that something.

    "Declared keyword names across every method of `f`, minus the `kwargs...` slurp."
    _declared_kwargs(f) = begin
        names = Symbol[]
        for m in methods(f)
            for k in Base.kwarg_decl(m)
                # the slurp shows up as `kwargs...`
                endswith(String(k), "...") && continue
                push!(names, k)
            end
        end
        Set(names)
    end

    @testset "kwarg validation — drift guard" begin
        # Entry points whose accepted set must COVER every keyword they declare.
        # (The table may be broader: a top-level sampler also accepts the kwargs of
        # the per-chain constructor it forwards into, which are not in its own
        # signature.)
        for (fname, f) in (
            :adaptive_warmup_mcmc    => adaptive_warmup_mcmc,
            :resume_warmup_mcmc      => resume_warmup_mcmc,
            :cooperative_warmup_mcmc => cooperative_warmup_mcmc,
            :clustered_warmup_mcmc   => clustered_warmup_mcmc,
            :clustered_chains        => clustered_chains,
        )
            declared = _declared_kwargs(f)
            accepted = Set(_SAMPLER_KWARGS[fname])
            # `parallel` is declared by adaptive's multi-chain method but must stay
            # REJECTED by its single-chain method — see the const's docstring. The
            # exemption is scoped here so the guard stays strict for the samplers
            # that genuinely declare and honour it.
            fname === :adaptive_warmup_mcmc && union!(accepted, _MULTICHAIN_ONLY_KWARGS)
            missing_from_table = setdiff(declared, accepted)
            @test isempty(missing_from_table)
            isempty(missing_from_table) || @info "kwargs declared but not in _SAMPLER_KWARGS" fname missing_from_table
        end

        # Every sampler that forwards to the initializer must offer the escape hatch.
        for fname in (:adaptive_warmup_mcmc, :cooperative_warmup_mcmc,
                      :clustered_warmup_mcmc, :clustered_chains)
            @test :pathfinder_kw in _SAMPLER_KWARGS[fname]
        end
    end

    # --- the check itself, without running a sampler ------------------------------
    @testset "kwarg validation — _check_kwargs" begin
        @test _check_kwargs(:adaptive_warmup_mcmc, pairs((;))) === nothing
        @test _check_kwargs(:adaptive_warmup_mcmc, pairs((; n_draws=10))) === nothing
        # initializer kwargs this package names itself are forwarded on purpose
        for k in _INITIALIZER_KWARGS
            @test _check_kwargs(:adaptive_warmup_mcmc, pairs(NamedTuple{(k,)}((1,)))) === nothing
        end
        @test_throws ArgumentError _check_kwargs(:adaptive_warmup_mcmc, pairs((; n_draw=10)))

        # the message must name the offender, and suggest the near miss
        msg = try
            _check_kwargs(:clustered_warmup_mcmc, pairs((; n_draw=10)))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("n_draw", msg)
        @test occursin("Did you mean `n_draws`", msg)
        @test occursin("pathfinder_kw", msg)

        # a capability difference points at the sampler that DOES support it
        msg = try
            _check_kwargs(:cooperative_warmup_mcmc, pairs((; callback=identity)))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("adaptive_warmup_mcmc", msg)

        # a foreign idiom names the WarmupHMC spelling
        msg = try
            _check_kwargs(:adaptive_warmup_mcmc, pairs((; initial_params=[1.0])))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("`init`", msg)
    end

    @testset "kwarg validation — edit distance" begin
        @test _edit_distance("n_draw", "n_draws") == 1
        @test _edit_distance("n_dwars", "n_draws") == 2
        @test _edit_distance("", "abc") == 3
        @test _edit_distance("abc", "abc") == 0
        @test _nearest_kwarg(:n_draw, (:n_draws, :n_evaluations)) === :n_draws
        # a genuinely unrelated name must NOT be "corrected" to something arbitrary
        @test _nearest_kwarg(:completely_unrelated, (:n_draws, :init)) === nothing
    end

    # --- end-to-end against the real samplers -------------------------------------
    struct KwargStdNormal
        d::Int
    end
    LogDensityProblems.dimension(p::KwargStdNormal) = p.d
    LogDensityProblems.capabilities(::Type{<:KwargStdNormal}) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.logdensity(p::KwargStdNormal, x) = -sum(abs2, x) / 2
    LogDensityProblems.logdensity_and_gradient(p::KwargStdNormal, x) = (-sum(abs2, x) / 2, -x)

    @testset "kwarg validation — end to end" begin
        d = 5
        lpdf = KwargStdNormal(d)
        # a NamedTuple `init` short-circuits Pathfinder, keeping these tests fast
        mkinit() = (; position = zeros(d), squared_scale = Matrix(1.0I, d, d) .+ 0.2)
        rngs() = [Xoshiro(i) for i in 1:2]

        # the four silent modes are now loud
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; n_draw=50, init=mkinit())
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; initial_params=zeros(d), init=mkinit())
        @test_throws ArgumentError adaptive_warmup_mcmc(Xoshiro(1), lpdf; parallel=false, init=mkinit())
        @test_throws ArgumentError cooperative_warmup_mcmc(rngs(), lpdf; n_evaluations_budget=5_000,
                                                           callback=(s, stage) -> false, init=mkinit())

        # legitimate calls are unaffected
        r = adaptive_warmup_mcmc(Xoshiro(1), lpdf; n_draws=50, init=mkinit())
        @test size(r.posterior_position, 2) >= 50

        # `parallel` IS legitimate on the multi-chain method
        rs = adaptive_warmup_mcmc(rngs(), lpdf; n_draws=50, parallel=false, init=mkinit())
        @test length(rs) == 2

        # deliberate Pathfinder passthrough is accepted and not mistaken for a typo
        r = adaptive_warmup_mcmc(Xoshiro(1), lpdf; n_draws=50, init=mkinit(),
                                 pathfinder_kw=(; some_pathfinder_option=1))
        @test size(r.posterior_position, 2) >= 50

        # clustered forwards per-chain kwargs it does not itself declare
        out = clustered_warmup_mcmc(rngs(), lpdf; n_draws=50, max_windows=2,
                                    regularizing_n=4.0, init=mkinit())
        @test length(out.results) == 2
        @test_throws ArgumentError clustered_warmup_mcmc(rngs(), lpdf; n_draws=50, max_windows=2,
                                                         regularizing_nn=4.0, init=mkinit())
    end
end
