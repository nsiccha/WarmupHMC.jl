# ── Entry-point keyword validation ──────────────────────────────────────────
#
# Every sampler entry point ends in `kwargs...` and forwards whatever it did not
# consume down to the initializer (`initialize_mcmc` → `mypathfinder` →
# `Pathfinder.pathfinder`), which absorbs anything. That makes four distinct
# consumer mistakes completely silent:
#
#   1. a kwarg this sampler cannot honour  — `callback` on the cooperative path
#      is accepted and never fires;
#   2. a typo                              — `n_draw` samples the default 1000
#      instead of the 100 that was asked for;
#   3. a neighbouring package's idiom      — `initial_params` (AdvancedHMC) is
#      accepted and ignored;
#   4. a kwarg belonging to a different METHOD of the same sampler — `parallel`
#      on the single-chain method vanishes.
#
# None of these can be detected by calling: the run completes and looks normal.
# The only way a consumer could establish the supported surface was runtime
# introspection of every method, which still cannot separate "supported" from
# "accepted and ignored".
#
# We deliberately do NOT try to whitelist the downstream surface exhaustively:
# `Pathfinder.pathfinder` itself ends in `kwargs...` and forwards into
# Optimization.jl, so the set of legal forwards is open-ended and not ours to
# enumerate — pinning it here would break on any Pathfinder version bump. The
# contract is instead:
#
#   * a kwarg the sampler (or the per-chain constructor it forwards into)
#     DECLARES is accepted — that set is the signature itself, so it stays
#     correct as signatures change;
#   * the small set of initializer kwargs this package names in its own source
#     is accepted (`_INITIALIZER_KWARGS`);
#   * anything else is an ERROR — with `pathfinder_kw = (; …)` as the explicit
#     escape hatch for deliberate passthrough.
#
# The accepted sets below are literal tuples rather than `Base.kwarg_decl`
# lookups so that include order does not matter and there is no per-call
# reflection. `test/test_kwarg_validation.jl` asserts each tuple still matches
# the live method signature, so a signature change that forgets this file fails
# CI instead of silently narrowing the accepted surface.

"""
Initializer keywords this package names in its own source, and therefore
forwards on purpose: `initialize_mcmc` (`ntries`, `maxiters`) and `mypathfinder`
(`ndraws`, `ndraws_elbo`, `history_length`, `optimizer`). Anything else destined
for Pathfinder goes through `pathfinder_kw`.
"""
const _INITIALIZER_KWARGS = (:ntries, :maxiters, :ndraws, :ndraws_elbo, :history_length, :optimizer)

# Keywords declared by the per-chain constructors the top-level samplers forward
# into. A top-level sampler accepts its own declared set PLUS the set of the
# constructor it feeds.
const _COOPERATIVE_CHAIN_KWARGS = (
    :n_draws, :n_evaluations, :recording_target, :stepsize_adaptation_limit,
    :target_acceptance_rate, :max_tree_depth, :init, :variance_cond_target,
    :nonlinear_adapt, :monitor_ess, :max_window_evaluations, :progress,
)
const _CLUSTERED_CHAIN_KWARGS = (
    :n_draws, :n_evaluations, :recording_target, :stepsize_adaptation_limit,
    :target_acceptance_rate, :max_tree_depth, :max_window_evaluations, :init,
    :regularizing_n, :regularizing_var, :weighting, :progress,
)

"""
Per-entry-point accepted keyword sets. Keys are the entry point's name as the
user spells it at the call site; values are every keyword that entry point can
actually honour. Used both to accept/reject and to tell a consumer which OTHER
sampler does support the keyword they reached for.
"""
const _SAMPLER_KWARGS = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :adaptive_warmup_mcmc => (
        # single-chain method
        :n_draws, :n_evaluations, :recording_target, :stepsize_adaptation_limit,
        :target_acceptance_rate, :max_tree_depth, :init, :progress, :description,
        :monitor_ess, :nonlinear_adapt, :variance_cond_target, :callback,
        :checkpoint_dir, :pathfinder_kw,
        # NB: `parallel` is deliberately absent — see `_MULTICHAIN_ONLY_KWARGS`.
    ),
    :resume_warmup_mcmc => (
        :progress, :description, :callback, :checkpoint_dir,
        :checkpoint_name, :parallel,
    ),
    :cooperative_warmup_mcmc => (
        :n_cores, :target_ess, :n_evaluations_budget, :time_budget, :min_chains,
        :max_window_evaluations, :n_draws, :nonlinear_adapt, :progress,
        :checkpoint_dir, :pathfinder_kw, _COOPERATIVE_CHAIN_KWARGS...,
    ),
    :clustered_warmup_mcmc => (
        :n_draws, :max_windows, :cluster_fn, :weighting, :metric, :threshold,
        :parallel, :n_evaluations_budget, :init, :progress, :checkpoint_dir,
        :pathfinder_kw,
        _CLUSTERED_CHAIN_KWARGS...,
    ),
    :clustered_chains => (
        :n_draws, :weighting, :init, :parallel, :progress, :pathfinder_kw,
        _CLUSTERED_CHAIN_KWARGS...,
    ),
)

"""
Keywords declared by `adaptive_warmup_mcmc`'s MULTI-CHAIN method but deliberately
kept OUT of its accepted set. They are consumed by the multi-chain signature and
never forwarded onward, so the single-chain method must still reject them —
`parallel=` on the single-chain method (where there is nothing to parallelise)
was one of the reported silent no-ops. The drift guard in
`test/kwarg_validation.jl` exempts these for that entry point only, so the guard
stays strict for the samplers that genuinely declare and honour `parallel`.
"""
const _MULTICHAIN_ONLY_KWARGS = (:parallel,)

"""
Keywords that are idiomatic in a NEIGHBOURING sampler package, or in a different
METHOD of this one, and therefore natural to reach for here. Mapping them
explicitly turns "silently ignored" into "here is the WarmupHMC spelling".
"""
const _FOREIGN_IDIOMS = Dict{Symbol,String}(
    :initial_params => "`init` — WarmupHMC takes the unconstrained starting point as `init` (AdvancedHMC spells it `initial_params`)",
    :n_adapts       => "`n_evaluations` (per-window gradient budget) and/or `stepsize_adaptation_limit`",
    :nadapts        => "`n_evaluations` (per-window gradient budget) and/or `stepsize_adaptation_limit`",
    :drop_warmup    => "nothing — WarmupHMC already returns post-warm-up draws",
    :discard_initial=> "nothing — WarmupHMC already returns post-warm-up draws",
    :n_chains       => "one `rng` per chain, positionally: `adaptive_warmup_mcmc(rngs, lpdf; …)`",
    :chains         => "one `rng` per chain, positionally: `adaptive_warmup_mcmc(rngs, lpdf; …)`",
    :nchains        => "one `rng` per chain, positionally: `adaptive_warmup_mcmc(rngs, lpdf; …)`",
    :seed           => "a seeded `rng`, positionally: `adaptive_warmup_mcmc(Xoshiro(1), lpdf; …)`",
    :rng            => "the `rng` is POSITIONAL: `adaptive_warmup_mcmc(rng, lpdf; …)`",
    :thin           => "nothing — WarmupHMC does not thin; subsample the returned draws yourself",
    :parallel       => "the MULTI-CHAIN method, which is where chains exist to run in parallel: `adaptive_warmup_mcmc(rngs, lpdf; parallel=…)`",
)

# Restricted Damerau-Levenshtein distance, for near-miss suggestions.
_edit_distance(a::AbstractString, b::AbstractString) = begin
    m, n = length(a), length(b)
    m == 0 && return n
    n == 0 && return m
    av, bv = collect(a), collect(b)
    prev2 = zeros(Int, n + 1)
    prev = collect(0:n)
    curr = zeros(Int, n + 1)
    for i in 1:m
        curr[1] = i
        for j in 1:n
            cost = av[i] == bv[j] ? 0 : 1
            curr[j+1] = min(curr[j] + 1, prev[j+1] + 1, prev[j] + cost)
            if i > 1 && j > 1 && av[i] == bv[j-1] && av[i-1] == bv[j]
                curr[j+1] = min(curr[j+1], prev2[j-1] + cost)   # transposition
            end
        end
        prev2, prev, curr = prev, copy(curr), prev2
    end
    prev[n+1]
end

# Closest accepted name to `k`, or `nothing` if nothing is close enough. The
# threshold scales with length so short names do not match everything.
_nearest_kwarg(k::Symbol, accepted) = begin
    ks = String(k)
    best, bestd = nothing, typemax(Int)
    for c in accepted
        d = _edit_distance(ks, String(c))
        d < bestd && ((best, bestd) = (c, d))
    end
    limit = max(1, min(3, length(ks) ÷ 3))
    bestd <= limit ? best : nothing
end

# Which OTHER entry points declare `k`? Lets the error say "supported by X, not
# here" for a real capability difference (`callback` on the cooperative path)
# rather than a generic "unknown keyword".
_supported_elsewhere(k::Symbol, fname::Symbol) =
    sort!([f for (f, ks) in _SAMPLER_KWARGS if f !== fname && k in ks])

_unknown_kwarg_message(fname::Symbol, k::Symbol, accepted) = begin
    io = IOBuffer()
    print(io, "`", fname, "` got an unsupported keyword argument `", k, "`.\n")

    near = _nearest_kwarg(k, accepted)
    isnothing(near) || print(io, "\n  Did you mean `", near, "`?\n")

    if haskey(_FOREIGN_IDIOMS, k)
        print(io, "\n  Instead, use ", _FOREIGN_IDIOMS[k], ".\n")
    end

    others = _supported_elsewhere(k, fname)
    isempty(others) || print(io, "\n  `", k, "` IS supported by ",
        join(string.("`", others, "`"), ", ", " and "), " — but not by `", fname, "`.\n")

    # A top-level sampler's accepted set is its own signature UNIONED with the
    # per-chain constructor's, so the two overlap — dedupe before displaying.
    shown = sort!(unique!(collect(String.(accepted))))
    print(io, """

    This is an error rather than a silent no-op because every WarmupHMC sampler
    forwards leftover keywords to the initializer, where an unrecognised name is
    absorbed without complaint: the run would complete normally and quietly not
    do what you asked.

    Accepted by `$fname`:
      $(join(shown, ", "))

    To forward a keyword to the Pathfinder initializer on purpose, pass it in
    `pathfinder_kw`, e.g. `pathfinder_kw = (; ndraws_elbo = 5)`.
    """)
    String(take!(io))
end

"""
    _check_kwargs(fname, kwargs)

Throw an `ArgumentError` naming the offender if `kwargs` carries a keyword
`fname` cannot honour. Called once per entry point, before any work.
"""
_check_kwargs(fname::Symbol, kwargs) = begin
    isempty(kwargs) && return nothing
    accepted = _SAMPLER_KWARGS[fname]
    for k in keys(kwargs)
        (k in accepted || k in _INITIALIZER_KWARGS) && continue
        throw(ArgumentError(_unknown_kwarg_message(fname, k, accepted)))
    end
    nothing
end
