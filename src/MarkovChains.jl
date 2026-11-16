# Latent Markov chains (random walks, AR(1) and OU series) whose states are
# centred adaptively per time step (snag random-walk-bloc-fc175e21, decision
# 1idvzrg). A chain is carried by an `IndexedReparametrization` next to its
# pairs and takes part in every path the pairs do: the map and its inverse, the
# gradient (one Enzyme reverse sweep per chain in `WarmupHMCEnzymeExt`), the
# window-end transport, selection, controls, snapshots and checkpoints.
#
# A chain whose readings see it relative to one of its own states takes that
# state as its `reference` (snag markovchain-on-a-ed98880f, decision 11fctf6):
# the walk then starts there and goes outward, and every other state's location
# gains a fitted share of the reference value.

"""
    MarkovChain(states; log_scale, coefficient = 1.0, drift = 0.0, initial = 0.0,
                reference = nothing, level = 0.0, phi = 0.0, c = 1.0, psi = 0.0)

A latent Markov chain, such as a random walk, an AR(1) series or a discretely
observed Ornstein–Uhlenbeck process, whose states warm-up centres adaptively,
with two controls per time step, three with a `reference`. Pass chains to an
[`IndexedReparametrization`](@ref) with `chains = [...]`.

`states` lists the chain's coordinates in time order, and the model is written in
the states `w_t` themselves. State `t`'s location `m_t` is `initial` for the
first state and `coefficient_t * w_{t-1} + drift_t` for every later one, with the
previous state read in the model's frame. The sampler works in

```
x_t = (w_t - phi_t * m_t) * exp((c_t - 1) * log_scale_t)
```

so `(phi_t, c_t) = (0, 1)` is the state itself (the model as written, centred)
and `(1, 0)` is the standardized innovation (non-centred) of a chain whose state
`t` given the previous one has mean `m_t` and standard deviation
`exp(log_scale_t)`. The log-Jacobian is `sum((1 - c_t) * log_scale_t)`.

At every restarting warm-up window, `phi_t` is refitted by least-squares
regression of `w_t` on `m_t` over the window's evidence, bounded to `[0, 1]`
(between the state and its innovation, where a Gaussian Markov posterior's
conditional slope lies), then `c_t` is chosen on the grid `range(0, 1, 11)` by
the default criterion of
[`candidate_scoring_losses`](@ref), the smallest position–gradient correlation,
with the gradients taken in the frame of the new `phi`. A chain thus ends up
centred where observations pin its states and close to the innovation where they
do not, such as in an unobserved stretch.

# Readings relative to one of the chain's states

When the model sees the chain only through its change since one of its own
states `w_r` (an anchor time), plus an offset, pass that state's number as
`reference = r` (in `1:length(states)`) and the offset as `level`: the readings
see state `t` as `level_t + (w_t - w_r)`. The regression on the previous state
alone cannot follow such a posterior, because every read state also depends on
`w_r`. With a reference:

- the walk starts at `r`, located at `initial`, and goes forward over
  `r+1, …, T` as above and backward over `r-1, …, 1`, where state `t` is
  located at its next state through that state's transition,
  `coefficient_{t+1} * w_{t+1} + drift_{t+1}`, with scale
  `log_scale_{t+1}`;
- every state but `r` gets a third control `psi_t`, the share of the reference
  value `R_t = w_r - level_t` in its location:

  ```
  x_t = (w_t - phi_t * m_t - psi_t * R_t) * exp((c_t - 1) * log_scale_t)
  ```

  with `m_t` and `log_scale_t` those of its place in the walk;
- at every restarting window `(phi_t, psi_t)` are refitted together by
  least-squares regression of `w_t` on `(m_t, R_t)`, bounded to the triangle
  `phi, psi >= 0`, `phi + psi <= 1` whose corners are the state, the state
  relative to the reference (`(0, 1)`, what the readings see) and the
  innovation; then `c_t` is chosen as above.

The backward stretch is exact for a random walk without drift and for a
stationary AR(1) or OU series; for a walk with a sampled trend its location
carries the drift with the sign as written, so it is approximate there. A
reference that the readings do not use costs nothing: `psi` then fits to zero.
The reference must be exactly the state the readings are relative to; the
consumer knows it from its likelihood.

# Arguments

`log_scale`, `coefficient`, `drift`, `initial` and `level` are each a `Real`
constant or a callable applied to the whole source position vector, as a
[`Reparametrization`](@ref)'s location and log-scale are. A callable may return
one number for every state, or a vector with one entry per state: a `log_scale`
whose first entry is the stationary scale of an AR(1) series, a `coefficient`
and `log_scale` for an OU process on an irregular time grid, or a `level` that
varies over time. The first entries of `coefficient` and `drift` are not read,
nor is `level` at the reference; `initial` is a number.

They may read only coordinates that no chain or pair transforms, such as a log
step size, an autoregression coefficient, a mean or a reading offset. They are
evaluated once per chain on every path, and they run under AD on every gradient
evaluation, so keep them cheap and type-generic, as for a `Reparametrization`.

`phi`, `c` and `psi` are the starting controls, one number for every state or a
vector with one entry per state; the default is the identity. `psi` needs a
reference and is zero there. `nonlinear_adapt = false` keeps them fixed.
Adaptation writes the controls in place: after an adaptive run, `chain.phi`,
`chain.c` and `chain.psi` hold them, and checkpoints record them under
`reparam_sources`.

# Examples

A random walk with a sampled log step size at coordinate 601, and an AR(1)
series around a mean, with a stationary first state:

```julia
walk = MarkovChain(1:600; log_scale = x -> x[601])
rho(x) = tanh(x[702])
ar1 = MarkovChain(602:700; coefficient = rho, drift = x -> (1 - rho(x)) * x[703],
    initial = x -> x[703],
    log_scale = x -> [x[701] - log1p(-rho(x)^2) / 2; fill(x[701], 98)])
ir = IndexedReparametrization(pairs; chains = [walk])
```

A walk whose readings are `base + (w_t - w_67)`, with the sampled offset `base`
at coordinate 602:

```julia
anchored = MarkovChain(1:600; log_scale = x -> x[601], reference = 67,
                       level = x -> x[602])
```
"""
struct MarkovChain{L,R,D,I,V}
    states::Vector{Int}
    log_scale::L
    coefficient::R
    drift::D
    initial::I
    # The reference state's number in `1:length(states)`, or 0 without one; then
    # `level` is `nothing` and `psi` is zero.
    reference::Int
    level::V
    phi::Vector{Float64}
    c::Vector{Float64}
    psi::Vector{Float64}
end

function MarkovChain(states; log_scale, coefficient=1.0, drift=0.0, initial=0.0,
                     reference=nothing, level=nothing, phi=0.0, c=1.0, psi=0.0)
    s = collect(Int, states)
    isempty(s) && throw(ArgumentError("a MarkovChain needs at least one state"))
    allunique(s) || throw(ArgumentError(
        "a MarkovChain's states must be distinct coordinates; got a repeated one in $s"))
    if isnothing(reference)
        isnothing(level) || throw(ArgumentError(
            "MarkovChain: `level` is the offset of readings relative to the `reference` " *
            "state; pass `reference` with it"))
        r = 0
    else
        reference isa Integer && !(reference isa Bool) && 1 <= reference <= length(s) ||
            throw(ArgumentError(
                "MarkovChain: `reference` is the number of one of the chain's states, in " *
                "1:$(length(s)); got $(repr(reference))"))
        r = Int(reference)
        isnothing(level) && (level = 0.0)
    end
    accessors = ((:log_scale, log_scale), (:coefficient, coefficient),
                 (:drift, drift), (:initial, initial))
    for (name, a) in (r == 0 ? accessors : (accessors..., (:level, level)))
        _is_accessor(a) || throw(ArgumentError(
            "MarkovChain: `$name` is a $(typeof(a)), which is neither a Real constant " *
            "nor callable. Pass a Real or a callable applied to the whole source " *
            "position vector, such as `x -> x[9]`."))
    end
    psi = _chain_controls(psi, length(s), :psi)
    r == 0 && any(!iszero, psi) && throw(ArgumentError(
        "MarkovChain: `psi` is the share of the reference state in each location; " *
        "pass `reference` with it"))
    r == 0 || iszero(psi[r]) || throw(ArgumentError(
        "MarkovChain: `psi` must be zero at the reference state $r"))
    MarkovChain(s, log_scale, coefficient, drift, initial, r, level,
                _chain_controls(phi, length(s), :phi), _chain_controls(c, length(s), :c), psi)
end

function _chain_controls(value, n, name)
    v = value isa Real ? fill(Float64(value), n) : collect(Float64, value)
    length(v) == n || throw(DimensionMismatch(
        "MarkovChain: `$name` has $(length(v)) entries for $n states"))
    all(isfinite, v) || throw(ArgumentError("MarkovChain: `$name` must be finite"))
    v
end

Base.copy(chain::MarkovChain) = MarkovChain(chain.states, chain.log_scale,
    chain.coefficient, chain.drift, chain.initial, chain.reference, chain.level,
    copy(chain.phi), copy(chain.c), copy(chain.psi))

_has_reference(chain::MarkovChain) = chain.reference != 0

# --- Validation, as part of `IndexedReparametrization`'s constructor -------------

_check_chains(pairs, ::Tuple{}) = nothing
function _check_chains(pairs, chains)
    chains isa Union{Tuple,AbstractVector} && all(c -> c isa MarkovChain, chains) ||
        throw(ArgumentError(
            "IndexedReparametrization: `chains` must be a vector or tuple of " *
            "MarkovChain; got a $(typeof(chains))"))
    chains isa AbstractVector && !isempty(chains) && !isconcretetype(eltype(chains)) &&
        throw(ArgumentError(
            "IndexedReparametrization chains must have a concrete element type " *
            "(got $(eltype(chains))): give the chains of one vector the same accessor " *
            "types, or pass chains of different types as a tuple"))
    owner = Dict{Int,Int}()
    for (n, chain) in enumerate(chains), idx in chain.states
        haskey(owner, idx) && throw(ArgumentError(
            "IndexedReparametrization: coordinate $idx is a state of chains " *
            "$(owner[idx]) and $n; a coordinate belongs to one chain"))
        owner[idx] = n
    end
    for (idx, _) in pairs
        idx isa Integer && haskey(owner, idx) && throw(ArgumentError(
            "IndexedReparametrization: coordinate $idx is a state of chain " *
            "$(owner[idx]) and also has a pair; a chain's states take no pairs"))
    end
    nothing
end

# --- The map ------------------------------------------------------------------
#
# Every path evaluates a chain's accessors once (`_chain_frames`) and walks its
# states in the walk order with `_chain_step` (source to model) or
# `_chain_unstep` (model to source), so the map, its inverse, the Enzyme
# objectives and the selection share one formula.
#
# The walk starts at the root `r` (the reference, or the first state without
# one), located at `initial`; then come `r+1, …, T`, each located at the state
# before it through its own transition, and `r-1, …, 1`, each located at the
# state after it through that state's transition. Without a reference the root
# is state 1 and the walk is the time order, the chain as it was before
# references, operation for operation.

@inline _chain_frames(chain::MarkovChain, x) = (
    maybecall(chain.log_scale, x), maybecall(chain.coefficient, x),
    maybecall(chain.drift, x), maybecall(chain.initial, x), _level_frame(chain.level, x))
@inline _level_frame(::Nothing, x) = nothing
@inline _level_frame(level, x) = maybecall(level, x)
@inline _per_state(value::Real, t) = value
@inline _per_state(value::AbstractVector, t) = value[t]

@inline _chain_root(chain::MarkovChain) = max(chain.reference, 1)
# The `j`-th state after the root in the walk order, `j` in `1:T-1`.
@inline _walk_state(r, T, j) = j <= T - r ? r + j : T - j
# The state a non-root state is located at, and the one it locates.
@inline _walk_neighbour(r, t) = t > r ? t - 1 : t + 1
@inline _walk_child(r, T, t) = t > r ? (t < T ? t + 1 : 0) : (t > 1 ? t - 1 : 0)
# The transition (the index into `coefficient`, `drift` and `log_scale`) that
# state `t` takes in the walk: its own forward and at the root, its next
# state's backward.
@inline _walk_transition(r, t) = t < r ? t + 1 : t
@inline _transition_location(frames, k, neighbour) =
    _per_state(frames[2], k) * neighbour + _per_state(frames[3], k)
# The reference value `R_t = w_r - level_t` in state `t`'s location, or nothing.
@inline _reference_term(::Nothing, wr, t) = nothing
@inline _reference_term(level, wr, t) = wr - _per_state(level, t)
@inline _shift(phi, location, psi, ::Nothing) = phi * location
@inline _shift(phi, location, psi, reference) = phi * location + psi * reference
# The model value of a state and its log-Jacobian term.
@inline function _chain_step(phi, psi, c, location, log_scale, reference, xt)
    log_step = (1 - c) * log_scale
    log_step, _shift(phi, location, psi, reference) + xt * exp(log_step)
end
# The source value of a state and its inverse log-Jacobian term.
@inline function _chain_unstep(phi, psi, c, location, log_scale, reference, yt)
    log_step = (1 - c) * log_scale
    -log_step, (yt - _shift(phi, location, psi, reference)) * exp(-log_step)
end

# Where the walks write their values (`nothing`: nowhere) and what they add to
# their total besides the log-Jacobian (`nothing`: nothing; `false` is a strong
# zero, so `log_step + false` is `log_step`, bit for bit).
@inline _record!(::Nothing, idx, value) = nothing
@inline _record!(out, idx, value) = (out[idx] = value; nothing)
@inline _seeded(::Nothing, idx, value) = false
@inline _seeded(seed, idx, value) = seed[idx] * value

# Source to model: the log-Jacobian plus `seed` times the model values, writing
# the model values into `y` (either may be `nothing`). The map, and the Enzyme
# extension's objective, which differentiates this.
function _chain_walk!(y, seed, chain, phi, psi, c, x, frames)
    states = chain.states
    T = length(states)
    r = _chain_root(chain)
    log_step, wr = _chain_step(phi[r], psi[r], c[r], frames[4], _per_state(frames[1], r),
                               nothing, x[states[r]])
    _record!(y, states[r], wr)
    total = zero(eltype(x)) + (log_step + _seeded(seed, states[r], wr))
    previous = wr
    for j in 1:T-1
        t = _walk_state(r, T, j)
        k = _walk_transition(r, t)
        neighbour = t == r - 1 ? wr : previous
        log_step, yt = _chain_step(phi[t], psi[t], c[t], _transition_location(frames, k, neighbour),
                                   _per_state(frames[1], k), _reference_term(frames[5], wr, t),
                                   x[states[t]])
        _record!(y, states[t], yt)
        total += log_step + _seeded(seed, states[t], yt)
        previous = yt
    end
    total
end

# Model to source: every location and the reference value are read off the model
# point `target`, so the states need no order. The inverse, and the Enzyme
# extension's inverse objective.
function _chain_unwalk!(source, seed, chain, phi, psi, c, target, frames)
    states = chain.states
    T = length(states)
    r = _chain_root(chain)
    wr = target[states[r]]
    log_step, sr = _chain_unstep(phi[r], psi[r], c[r], frames[4], _per_state(frames[1], r),
                                 nothing, wr)
    _record!(source, states[r], sr)
    total = zero(eltype(target)) + (log_step + _seeded(seed, states[r], sr))
    for j in 1:T-1
        t = _walk_state(r, T, j)
        k = _walk_transition(r, t)
        location = _transition_location(frames, k, target[states[_walk_neighbour(r, t)]])
        log_step, st = _chain_unstep(phi[t], psi[t], c[t], location, _per_state(frames[1], k),
                                     _reference_term(frames[5], wr, t), target[states[t]])
        _record!(source, states[t], st)
        total += log_step + _seeded(seed, states[t], st)
    end
    total
end

_is_identity(chain::MarkovChain) =
    all(iszero, chain.phi) && all(isone, chain.c) && all(iszero, chain.psi)

# Chains of different types come as a tuple. Every path that walks them goes
# through a tuple element by element (recursion, or `map`, unrolled at compile
# time), so each chain's code is type-stable, as AD needs; a vector of chains has
# one concrete type and is looped over.
_each_chain(f, chains::Tuple) = (map(f, chains); nothing)
_each_chain(f, chains) = foreach(f, chains)
_fold_chains(f, value, ::Tuple{}) = value
_fold_chains(f, value, chains::Tuple) = _fold_chains(f, f(value, first(chains)), Base.tail(chains))
function _fold_chains(f, value, chains)
    for chain in chains
        value = f(value, chain)
    end
    value
end

# Write the chains' states of `y` from `x` (source to model) and add their
# log-Jacobian to `ljac`.
_chains_forward!(ljac, y, chains, x) =
    _fold_chains((ljac, chain) -> _chain_forward_or_copy!(ljac, y, chain, x), ljac, chains)
function _chain_forward_or_copy!(ljac, y, chain, x)
    if _is_identity(chain)
        for idx in chain.states
            y[idx] = x[idx]
        end
        ljac
    else
        ljac + _chain_walk!(y, nothing, chain, chain.phi, chain.psi, chain.c, x,
                            _chain_frames(chain, x))
    end
end

# Recover the chains' states of `source` from `target` (model to source). Their
# accessors read coordinates nothing transforms, the same in either frame; they
# are read off `target`.
_chains_inverse!(ljac, source, chains, target) =
    _fold_chains((ljac, chain) -> _chain_inverse_or_copy!(ljac, source, chain, target), ljac, chains)
function _chain_inverse_or_copy!(ljac, source, chain, target)
    if _is_identity(chain)
        for idx in chain.states
            source[idx] = target[idx]
        end
        ljac
    else
        ljac + _chain_unwalk!(source, nothing, chain, chain.phi, chain.psi, chain.c, target,
                              _chain_frames(chain, target))
    end
end

# --- Controls --------------------------------------------------------------------
#
# `(; phi, c)` per chain, and `(; reference, phi, c, psi)` for a chain with a
# reference, whose `phi` is relative to the walk that starts there.

_chain_controls(chain::MarkovChain) = _has_reference(chain) ?
    (; reference=chain.reference, phi=copy(chain.phi), c=copy(chain.c), psi=copy(chain.psi)) :
    (; phi=copy(chain.phi), c=copy(chain.c))
function _restore_chain_controls!(chain::MarkovChain, controls, n)
    length(controls.phi) == length(controls.c) == length(chain.states) ||
        throw(ArgumentError(
            "chain $n has $(length(chain.states)) states but the controls recorded " *
            "$(length(controls.phi)); build the chains exactly as the run was given them"))
    recorded = haskey(controls, :reference) ? controls.reference : 0
    recorded == chain.reference || throw(ArgumentError(
        "chain $n has " * (_has_reference(chain) ? "reference state $(chain.reference)" : "no reference") *
        " but the controls were recorded " *
        (recorded == 0 ? "without one" : "with reference state $recorded") *
        "; build the chains exactly as the run was given them"))
    chain.phi .= controls.phi
    chain.c .= controls.c
    _has_reference(chain) && (chain.psi .= controls.psi)
    chain
end
_snapshot_chains(chains::Tuple) = map(copy, chains)
_snapshot_chains(chains::AbstractVector) = map(copy, chains)
_restore_chains!(chains, old) =
    foreach((chain, o) -> (chain.phi .= o.phi; chain.c .= o.c; chain.psi .= o.psi), chains, old)

# --- Selection ---------------------------------------------------------------------
#
# The evidence is a set of weighted source-frame columns with their gradients. For
# a column, the chain's model values `w_t`, locations `m_t` and reference values
# `R_t` follow from the current controls `(phi0, psi0, c0)`. The gradient of state
# `t`'s source coordinate is `a_t * exp((1 - c0_t) * log_scale_t)`, where `a_t` is
# the derivative in `w_t` through every state the walk reaches from `t`,
#
#     a_t = v_t + phi0_s * coefficient_s * a_s           (s the state `t` locates)
#               + [t == r] * sum(psi0_u * a_u, u != r)   (the reference's share)
#
# and `v_t` is the model's own derivative in `w_t`. Undoing that recursion gives
# `v`; redoing it with the new `(phi, psi)` gives the gradient in the new frame,
# which is what the candidates for `c` are scored on. Each column costs one pass
# over the chain per direction, so a selection is linear in the chain's length.

const CHAIN_CENTERINGS = range(0, 1, 11)

_column_weight(::Nothing, k) = 1.0
_column_weight(weights, k) = Float64(weights[k])

# A state's derivative in `w_t` under the controls `c0` the column was drawn in.
@inline _column_adjoint(g, chain, c0, frames, r, t) =
    g[chain.states[t]] * exp(-((1 - c0[t]) * _per_state(frames[1], _walk_transition(r, t))))

# Model values `w`, locations `m` and reference values `R` (zero without a
# reference) of one column under `(phi0, psi0, c0)`, and the derivative `a` in
# `w` through every state the walk reaches under `(phi, psi)`.
function _chain_column!(w, m, R, a, chain, phi0, psi0, c0, phi, psi, x, g, frames)
    states = chain.states
    T = length(states)
    r = _chain_root(chain)
    m[r] = frames[4]
    R[r] = 0.0
    _, w[r] = _chain_step(phi0[r], psi0[r], c0[r], m[r], _per_state(frames[1], r), nothing,
                          x[states[r]])
    for j in 1:T-1
        t = _walk_state(r, T, j)
        k = _walk_transition(r, t)
        m[t] = _transition_location(frames, k, w[_walk_neighbour(r, t)])
        reference = _reference_term(frames[5], w[r], t)
        R[t] = something(reference, 0.0)
        _, w[t] = _chain_step(phi0[t], psi0[t], c0[t], m[t], _per_state(frames[1], k),
                              reference, x[states[t]])
    end
    # The reverse of the walk: each state after the one it locates.
    shares = 0.0
    for j in T-1:-1:1
        t = _walk_state(r, T, j)
        a_old = _column_adjoint(g, chain, c0, frames, r, t)
        child = _walk_child(r, T, t)
        a[t] = child == 0 ? a_old :
            a_old - (phi0[child] * _column_adjoint(g, chain, c0, frames, r, child) -
                     phi[child] * a[child]) * _per_state(frames[2], _walk_transition(r, child))
        _has_reference(chain) && (shares += psi0[t] * a_old - psi[t] * a[t])
    end
    a_root = _column_adjoint(g, chain, c0, frames, r, r)
    for child in (r + 1, r - 1)
        1 <= child <= T || continue
        a_root -= (phi0[child] * _column_adjoint(g, chain, c0, frames, r, child) -
                   phi[child] * a[child]) * _per_state(frames[2], _walk_transition(r, child))
    end
    a[r] = _has_reference(chain) ? a_root - shares : a_root
    nothing
end

# Weighted moments of `(w_t, m_t, R_t)` over a window's columns, for the
# least-squares fit of `(phi_t, psi_t)`: centred co-moments, updated as
# `WeightedReparametrizationLoss` updates its own (Welford, weighted).
mutable struct _ChainRegression
    weight::Float64
    mean_w::Float64
    mean_m::Float64
    mean_R::Float64
    mm::Float64
    RR::Float64
    mR::Float64
    wm::Float64
    wR::Float64
end
_ChainRegression() = _ChainRegression(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
function _observe_regression!(s::_ChainRegression, w, m, R, weight)
    new_weight = s.weight + weight
    fraction = weight / new_weight
    dw, dm, dR = w - s.mean_w, m - s.mean_m, R - s.mean_R
    s.mean_w += fraction * dw
    s.mean_m += fraction * dm
    s.mean_R += fraction * dR
    s.mm += weight * dm * (m - s.mean_m)
    s.RR += weight * dR * (R - s.mean_R)
    s.mR += weight * dm * (R - s.mean_R)
    s.wm += weight * dw * (m - s.mean_m)
    s.wR += weight * dw * (R - s.mean_R)
    s.weight = new_weight
    s
end

# The slope of `w_t` on `m_t`, bounded to [0, 1], between the state (0) and its
# innovation (1). A Gaussian Markov posterior's conditional slope lies there: it
# is the prior transition's share q⁻² / (q⁻² + λ_t) of the state's precision.
# Unbounded, a location that barely varies over the window gives a slope of noise
# over a vanishing variance: an Ornstein-Uhlenbeck coefficient exp(-1 / range) at
# a tiny range made m_t ≈ 1e-65 and phi 1e47–1e67, and at an ordinary range
# phi * m_t then moved every state so far that the chain never returned (snag
# non-centred-ou-s-83c75167). `phi` keeps its value where `m_t` does not vary.
function _fit_slope(s::_ChainRegression, phi0)
    slope = s.wm / s.mm
    s.mm > 0 && isfinite(slope) ? clamp(slope, 0.0, 1.0) : phi0
end

# `(phi, psi)` minimizing the residual variance of `w - phi * m - psi * R` on the
# triangle `phi, psi >= 0`, `phi + psi <= 1`. Its corners are the state `(0, 0)`,
# the state relative to the reference `(0, 1)` and the innovation `(1, 0)`. The
# variance is a convex quadratic, so its minimum is the unconstrained one when
# that lies inside, and otherwise the best of the three edges' minima. A control
# whose regressor does not vary is not identified and keeps its value (clamped
# into the triangle), the other one being fitted beside it.
function _fit_triangle(s::_ChainRegression, phi0, psi0)
    # The variance less Var(w), at (p, q).
    objective(p, q) = p * (p * s.mm - 2 * s.wm) + q * (q * s.RR - 2 * s.wR) + 2 * p * q * s.mR
    edge(numerator, denominator, upper) =
        denominator > 0 && isfinite(numerator / denominator) ?
            clamp(numerator / denominator, 0.0, upper) : nothing
    if !(s.mm > 0) && !(s.RR > 0)
        return phi0, psi0
    elseif !(s.mm > 0)
        p = clamp(phi0, 0.0, 1.0)
        return p, something(edge(s.wR, s.RR, 1 - p), clamp(psi0, 0.0, 1 - p))
    elseif !(s.RR > 0)
        q = clamp(psi0, 0.0, 1.0)
        return something(edge(s.wm, s.mm, 1 - q), clamp(phi0, 0.0, 1 - q)), q
    end
    determinant = s.mm * s.RR - s.mR^2
    if determinant > 1e-12 * s.mm * s.RR
        p = (s.RR * s.wm - s.mR * s.wR) / determinant
        q = (s.mm * s.wR - s.mR * s.wm) / determinant
        isfinite(p) && isfinite(q) && p >= 0 && q >= 0 && p + q <= 1 && return p, q
    end
    # The edges psi = 0, phi = 0 and phi + psi = 1 (where w - R = phi * (m - R)).
    candidates = Tuple{Float64,Float64}[]
    p = edge(s.wm, s.mm, 1.0)
    isnothing(p) || push!(candidates, (p, 0.0))
    q = edge(s.wR, s.RR, 1.0)
    isnothing(q) || push!(candidates, (0.0, q))
    p = edge(s.wm - s.wR - s.mR + s.RR, s.mm - 2 * s.mR + s.RR, 1.0)
    isnothing(p) || push!(candidates, (p, 1 - p))
    isempty(candidates) && return phi0, psi0
    candidates[argmin([objective(pq...) for pq in candidates])]
end

# The fitted `(phi, psi)` of every state, and the candidates' loss accumulators for
# `c_t` in the frame of those controls. Returns `(phi, psi, losses, groups)`:
# `losses[j, t]` is candidate `CHAIN_CENTERINGS[j]` of state `t`, and `groups` the
# number of positive-weight columns.
function _chain_scores(chain, positions, gradients, weights)
    T = length(chain.states)
    r = _chain_root(chain)
    phi0, psi0, c0 = chain.phi, chain.psi, chain.c
    w, m, R, a = zeros(T), zeros(T), zeros(T), zeros(T)
    columns = [k for k in axes(positions, 2) if _column_weight(weights, k) > 0]
    regression = [_ChainRegression() for _ in 1:T]
    for k in columns
        x = view(positions, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, R, a, chain, phi0, psi0, c0, phi0, psi0, x, view(gradients, :, k), frames)
        for t in 1:T
            _observe_regression!(regression[t], w[t], m[t], R[t], _column_weight(weights, k))
        end
    end
    phi, psi = copy(phi0), copy(psi0)
    for t in 1:T
        if _has_reference(chain) && t != r
            phi[t], psi[t] = _fit_triangle(regression[t], phi0[t], psi0[t])
        else
            phi[t] = _fit_slope(regression[t], phi0[t])
        end
    end
    losses = [WeightedReparametrizationLoss() for _ in CHAIN_CENTERINGS, _ in 1:T]
    for k in columns
        x = view(positions, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, R, a, chain, phi0, psi0, c0, phi, psi, x, view(gradients, :, k), frames)
        weight = _column_weight(weights, k)
        for t in 1:T
            log_scale = _per_state(frames[1], _walk_transition(r, t))
            residual = w[t] - _shift(phi[t], m[t], psi[t], _has_reference(chain) ? R[t] : nothing)
            # The candidates' scales exp((1 - c) * log_scale) differ by one factor
            # per grid step: two `exp`s per state rather than two per candidate.
            factor = exp(step(CHAIN_CENTERINGS) * log_scale)
            scale = exp((1 - first(CHAIN_CENTERINGS)) * log_scale)
            for (j, c) in enumerate(CHAIN_CENTERINGS)
                OnlineStatsBase.fit!(losses[j, t],
                    ((1 - c) * log_scale, residual / scale, a[t] * scale); weight)
                scale /= factor
            end
        end
    end
    phi, psi, losses, length(columns)
end

# Choose the chain's new controls from the evidence, as `_minimizing_source!` does
# for a pair: only finite losses compete, a state with none keeps its centering,
# and `excluded` collects `index => kept` for `_warn_excluded_candidates`. Returns
# whether the controls changed.
function _select_chain!(excluded, chain, positions, gradients, weights=nothing;
                        loss_kwargs...)
    phi, psi, losses, groups = _chain_scores(chain, positions, gradients, weights)
    groups > 2 || return false
    c = copy(chain.c)
    for t in eachindex(c)
        values = [reparametrization_loss(losses[j, t]; loss_kwargs...)
                  for j in eachindex(CHAIN_CENTERINGS)]
        finite = any(isfinite, values)
        any(!isfinite, values) && push!(excluded, chain.states[t] => !finite)
        finite || continue
        c[t] = CHAIN_CENTERINGS[argmin(j -> isfinite(values[j]) ? values[j] : Inf,
                                       eachindex(values))]
    end
    changed = phi != chain.phi || c != chain.c || psi != chain.psi
    chain.phi .= phi
    chain.c .= c
    chain.psi .= psi
    changed
end

# Move the chain's states of every pool column from `(phi0, psi0, c0)` into the
# chain's current controls, positions and gradients, so that what later pairs
# score reads them in the new frame, as `_select_on_pool!` moves its pairs'
# coordinates.
function _move_chain_columns!(chain, phi0, psi0, c0, positions, gradients)
    T = length(chain.states)
    r = _chain_root(chain)
    w, m, R, a = zeros(T), zeros(T), zeros(T), zeros(T)
    for k in axes(positions, 2)
        x, g = view(positions, :, k), view(gradients, :, k)
        frames = _chain_frames(chain, x)
        _chain_column!(w, m, R, a, chain, phi0, psi0, c0, chain.phi, chain.psi, x, g, frames)
        for (t, idx) in enumerate(chain.states)
            log_step = (1 - chain.c[t]) * _per_state(frames[1], _walk_transition(r, t))
            reference = _has_reference(chain) ? R[t] : nothing
            x[idx] = (w[t] - _shift(chain.phi[t], m[t], chain.psi[t], reference)) * exp(-log_step)
            g[idx] = a[t] * exp(log_step)
        end
    end
    nothing
end

# The pool search's chain step, before the pairs: each chain is selected on the
# pool and its states are moved into the new frame.
_select_chains_on_pool!(::Tuple{}, xg...; loss_kwargs...) = nothing
function _select_chains_on_pool!(chains, positions, gradients; loss_kwargs...)
    excluded = Pair{Any,Bool}[]
    for chain in chains
        phi0, psi0, c0 = copy(chain.phi), copy(chain.psi), copy(chain.c)
        _select_chain!(excluded, chain, positions, gradients; loss_kwargs...) &&
            _move_chain_columns!(chain, phi0, psi0, c0, positions, gradients)
    end
    _warn_excluded_candidates(excluded)
    nothing
end

# --- Streaming evidence -------------------------------------------------------------
#
# A chain's candidates are scored in the frame of the `phi` the window ends with,
# which the leaves cannot know as they stream in, so a chain keeps a weighted
# sample of the window's leaves (the evidence a `WindowSelectionPlan` gets) and is
# selected on it at the window's end. The pairs keep their running accumulators.

struct ChainEvidence{O}
    pairs::O
    sample::WeightedLeafSample
end

_observe_nonlinear!(plan::Union{DirectCandidateScoring,CandidateScoringPlan}, ir,
                    online::ChainEvidence, position, gradient, weight) = begin
    _observe_nonlinear!(plan, ir, online.pairs, position, gradient, weight)
    _observe_sample!(online.sample, position, gradient, weight)
    nothing
end
_mark_group!(online::ChainEvidence) = _mark_group!(online.pairs)
reset!(online::ChainEvidence) = (reset!(online.pairs); reset!(online.sample); nothing)
_candidate_scoring_losses(online::ChainEvidence) = _candidate_scoring_losses(online.pairs)
_leaf_sample(online::ChainEvidence) = online.sample
_leaf_sample(online::WeightedLeafSample) = online
_leaf_sample(online) = nothing

function optimize!(ir::IndexedReparametrization, online::ChainEvidence; loss_kwargs=(;))
    excluded = Pair{Any,Bool}[]
    positions, gradients = _sample_evidence(online.sample)
    for chain in ir.chains
        _select_chain!(excluded, chain, positions, gradients; loss_kwargs...)
    end
    _warn_excluded_candidates(excluded)
    optimize!(ir, online.pairs; loss_kwargs)
end

# --- Retrospective loss rows ----------------------------------------------------------

function _chain_candidate_scoring_losses(chains, positions, gradients, weights)
    isnothing(weights) || for weight in weights
        weight >= 0 || throw(ArgumentError(
            "nonlinear observation weight must be nonnegative, got $weight"))
    end
    rows = NamedTuple[]
    for (chain_number, chain) in enumerate(chains)
        phi, psi, losses, groups = _chain_scores(chain, positions, gradients, weights)
        for (t, index) in enumerate(chain.states), (j, c) in enumerate(CHAIN_CENTERINGS)
            accumulator = losses[j, t]
            push!(rows, (; chain_number, state_number=t, index, phi=phi[t], psi=psi[t], candidate=c,
                loss=groups > 2 ? reparametrization_loss(accumulator) : missing,
                groups,
                effective_n=iszero(accumulator.weight2) ? missing : effective_n(accumulator)))
        end
    end
    rows
end
