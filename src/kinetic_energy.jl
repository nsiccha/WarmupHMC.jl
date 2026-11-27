# ------------------------------------------------------- per-leapfrog metric work
#
# Every sampler stores its metric as DynamicHMC's
# `GaussianKineticEnergy(M⁻¹ = L L', W = L'⁻¹)` built from a scale `L`
# (`_stream_kinetic_energy`, `_coop_energy`, the adaptive `energy_options`).
# DynamicHMC applies `M⁻¹` three times per leapfrog step: to the half-step
# momentum in `leapfrog`, and twice to the new momentum, for the leaf's kinetic
# energy (`logdensity`) and for its turn statistic (`calculate_p♯`). With
# WarmupHMC's structured scales one `M⁻¹` is two operator products, each
# O(d · k) for the Pathfinder and reflection scales, so a step cost six of them:
# more than the gradient itself on a 1,000- to 2,000-dimensional logistic
# regression (snag `fixed-kernel-nut-d274a5f0`).
#
# `TrajectoryEnergy` wraps that kinetic energy in a `Hamiltonian` and carries
# the products along the trajectory, so a leapfrog step costs ONE `M⁻¹`
# product:
#   * it keeps, for the last leaf reached in each direction, its momentum `p`
#     with `p♯ = M⁻¹ p`, and its gradient `∇ℓ` with `g♯ = M⁻¹ ∇ℓ`;
#   * the step's `M⁻¹ pₘ` is `p♯ + ϵ/2 g♯` (linearity, no product), the new
#     point's `g♯′ = M⁻¹ ∇ℓ′` is the step's one product, and the new leaf's
#     `p♯′ = M⁻¹ pₘ + ϵ/2 g♯′`;
#   * the leaf's kinetic energy `dot(p′, p♯′) / 2` and its turn statistic read
#     that `p♯′`.
# A vector the wrapper does not hold (a transition's random start momentum, the
# gradient a transition starts from, the selected leaf's final kinetic energy) is
# multiplied directly and kept in a third slot, so a transition adds two or three
# products to its one per step. The keys are the vectors themselves (`===`):
# DynamicHMC never mutates a momentum or gradient after creating it, and the
# wrapper holds every key it compares, so no address is reused while it is held.
#
# `p♯′` and `M⁻¹ pₘ` equal the direct products in exact arithmetic and differ
# from them by rounding, so draws differ from a direct-product run by rounding
# too; the kernel is the same. A divergent leaf (non-finite log density) gets no
# product: its gradient is never read, as before.
#
# Build one per transition (`_hamiltonian` for each `sample_tree` call, or for
# each initial step-size search), never one per window: a carried product read
# from a slot and the same product computed directly differ by rounding, so a
# wrapper that outlived its transition would make a draw depend on the
# transitions before it, and a resumed run would no longer reproduce the
# uninterrupted one bit for bit. The stored kinetic energy stays free of
# scratch space, so states sharing it may still be stepped concurrently.

mutable struct TrajectoryEnergy{K<:DynamicHMC.GaussianKineticEnergy,V<:AbstractVector} <: DynamicHMC.EuclideanKineticEnergy
    κ::K
    # slot 1: the last leaf reached forward; 2: backward; 3: the last vector
    # multiplied directly
    p::Vector{Union{Nothing,V}}
    p♯::Vector{Union{Nothing,V}}
    g::Vector{Union{Nothing,V}}
    g♯::Vector{Union{Nothing,V}}
    # The transition's leapfrog points where the target answered a non-finite
    # log density other than -Inf, or a finite one with a non-finite gradient
    # (`_evaluate_leaf!`), and the first one's log density and up to five of its
    # non-finite gradient coordinates.
    nonfinite::Int
    first_nonfinite::Union{Nothing,Tuple{Float64,Vector{Int}}}
end
TrajectoryEnergy(κ::DynamicHMC.GaussianKineticEnergy) = begin
    V = Vector{eltype(κ.M⁻¹)}
    slots() = Union{Nothing,V}[nothing, nothing, nothing]
    TrajectoryEnergy{typeof(κ),V}(κ, slots(), slots(), slots(), slots(), 0, nothing)
end

# The Hamiltonian a sampler builds for its transitions. Only the Gaussian
# kinetic energy WarmupHMC stores is wrapped.
_trajectory_energy(κ::DynamicHMC.GaussianKineticEnergy) = TrajectoryEnergy(κ)
_trajectory_energy(κ) = κ
_hamiltonian(κ, lpdf) = DynamicHMC.Hamiltonian(_trajectory_energy(κ), lpdf)

Base.size(e::TrajectoryEnergy, args...) = size(e.κ, args...)
Base.show(io::IO, e::TrajectoryEnergy) = show(io, e.κ)

_slot(keys, x) = begin
    for i in eachindex(keys)
        keys[i] === x && return i
    end
    0
end
# `M⁻¹ x` for a momentum (`keys = e.p`) or a gradient (`keys = e.g`): read from
# a slot, or multiplied directly and kept in slot 3.
_carried_product!(e::TrajectoryEnergy{K,V}, keys, values, x) where {K,V} = begin
    i = _slot(keys, x)
    i == 0 || return something(values[i])  # a stored vector key always has a product
    v = e.κ.M⁻¹ * x
    if x isa V && v isa V
        keys[3] = x
        values[3] = v
    end
    v
end

DynamicHMC.calculate_p♯(e::TrajectoryEnergy, p, q=nothing) = _carried_product!(e, e.p, e.p♯, p)
DynamicHMC.∇kinetic_energy(e::TrajectoryEnergy, p, q=nothing) = DynamicHMC.calculate_p♯(e, p)
DynamicHMC.kinetic_energy(e::TrajectoryEnergy, p, q=nothing) = dot(p, DynamicHMC.calculate_p♯(e, p)) / 2
DynamicHMC.rand_p(rng::Random.AbstractRNG, e::TrajectoryEnergy, q=nothing) = DynamicHMC.rand_p(rng, e.κ)

# DynamicHMC's leapfrog with the carried products. `(p, ∇ℓ)` of the point it
# starts from are normally the last leaf in this direction (or the
# transition's start, kept in slot 3), so `p♯` and `g♯` are read, not computed.
function DynamicHMC.leapfrog(H::DynamicHMC.Hamiltonian{<:TrajectoryEnergy{K,V}},
        z::DynamicHMC.PhasePoint, ϵ) where {K,V}
    (; ℓ, κ) = H
    (; p, Q) = z
    isfinite(Q.ℓq) ||
        throw(ArgumentError("Internal error: leapfrog called from non-finite log density"))
    p♯ = _carried_product!(κ, κ.p, κ.p♯, p)
    g♯ = _carried_product!(κ, κ.g, κ.g♯, Q.∇ℓq)
    pₘ = p .+ (ϵ / 2) .* Q.∇ℓq
    p♯ₘ = p♯ .+ (ϵ / 2) .* g♯                  # M⁻¹ pₘ
    q′ = Q.q .+ ϵ .* p♯ₘ
    Q′ = _evaluate_leaf!(κ, ℓ, q′)
    pₘ .+= (ϵ / 2) .* Q′.∇ℓq                    # p′
    if isfinite(Q′.ℓq)
        g♯′ = κ.κ.M⁻¹ * Q′.∇ℓq
        p♯ₘ .+= (ϵ / 2) .* g♯′                  # M⁻¹ p′
        if pₘ isa V && p♯ₘ isa V && Q′.∇ℓq isa V && g♯′ isa V
            k = ϵ > 0 ? 1 : 2
            κ.p[k] = pₘ
            κ.p♯[k] = p♯ₘ
            κ.g[k] = Q′.∇ℓq
            κ.g♯[k] = g♯′
        end
    end
    DynamicHMC.PhasePoint(Q′, pₘ)
end

# ------------------------------------------------------- non-finite target answers
#
# A leapfrog point where the target answers `+Inf` or `NaN`, or a finite log
# density with a non-finite gradient, becomes a divergent leaf exactly as
# DynamicHMC's `evaluate_ℓ` makes it: its log density is replaced by `-Inf` and
# its gradient is never read, so the draws do not change. Unlike `-Inf`, the
# answer for a point outside the support, it is never legitimate (warmuphmc-use
# §1a), and as a plain divergence it is invisible: in snag
# `hierarchical-spa-1a7808bd` Stan's linear-space `simplex` underflowed to 0,
# `dirichlet` answered `+Inf`, and the run showed only 5.9–11% divergences. So
# the leapfrog counts these answers, every sampler reports the count as
# `n_nonfinite_leaves` (the one-step entry points as `nonfinite`), and a run warns
# at its first one.
function _evaluate_leaf!(e::TrajectoryEnergy, ℓ, q)
    all(isfinite, q) || return DynamicHMC.evaluate_ℓ(ℓ, q)   # DynamicHMC's own error
    ℓq, ∇ℓq = LogDensityProblems.logdensity_and_gradient(ℓ, q)
    ((isfinite(ℓq) && all(isfinite, ∇ℓq)) || ℓq == -Inf) &&
        return DynamicHMC.EvaluatedLogDensity(q, ℓq, ∇ℓq)
    e.nonfinite += 1
    isnothing(e.first_nonfinite) &&
        (e.first_nonfinite = (Float64(ℓq), first(findall(!isfinite, ∇ℓq), 5)))
    DynamicHMC.EvaluatedLogDensity(q, oftype(ℓq, -Inf), ∇ℓq)
end

# The non-finite answers of the transition `H` was built for (`_hamiltonian`).
_nonfinite_leaves(H::DynamicHMC.Hamiltonian) = H.κ isa TrajectoryEnergy ? H.κ.nonfinite : 0

# Adds a transition's non-finite answers to a run's `total` and warns at the
# run's first one; returns the new total.
function _take_nonfinite!(H::DynamicHMC.Hamiltonian, total::Integer)
    n = _nonfinite_leaves(H)
    n == 0 && return total
    if total == 0
        ℓq, coordinates = H.κ.first_nonfinite
        @warn "The target answered a non-finite log density (+Inf or NaN), or a finite " *
              "one with a non-finite gradient, at a leapfrog point (first: log density " *
              "$ℓq, non-finite gradient coordinates $coordinates). It is counted as a " *
              "divergence; `n_nonfinite_leaves` counts these answers. -Inf is the answer " *
              "for a point outside the target's support; any other non-finite answer " *
              "is a defect of the target, such as an underflow or overflow."
    end
    total + n
end
