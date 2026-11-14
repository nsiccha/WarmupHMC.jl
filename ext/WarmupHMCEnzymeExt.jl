module WarmupHMCEnzymeExt

# Reverse-mode Enzyme gradients of an `IndexedReparametrization`, run by run.
#
# Differentiating the whole transform in one `autodiff` call made Enzyme
# allocate a tape for every accessor call it reached through the loop over
# pairs: 19.4 MiB per gradient on a BRM-shaped correlated block of 26,400 pairs,
# against 0.2 MiB for evaluating the map (snag `find-reparametri-e2a5b2a7`). One
# `autodiff` call per pair allocates nothing. Its tape is that pair's alone.
# The map that evaluates accessors once per run of pairs (snag
# `use-the-public-i-0660224b`) no longer allocates per pair in one pass on that
# block either: three working vectors instead of 548 bytes per pair, measured
# with `test/pairwise_enzyme_gradient.jl`'s correlated block.
#
# Pairs that share accessor work (a run, `WarmupHMC._run_end`) are differentiated
# together, one `autodiff` per run, so their shared location or log-scale is
# evaluated and differentiated once rather than once per pair (snag
# `use-the-public-i-0660224b`). A run's indices increase, so each of its pairs
# takes its own coordinate's seed. A run of one pair is the pair-by-pair call.
# A pair whose source and target centerings are equal is the identity on its
# coordinate: it takes no call, its derivative is its seed, and a run of such
# pairs evaluates no accessor (snag `centering-evalua-54d5e611`).
#
# The map reads each run's shared values ("frames") off its partly mapped output:
# earlier runs at their model values, everything else at the sampler's (snag
# `indexedreparamet-1e1f6ef7`). A run's frames read no coordinate of their own run
# or of a later one, which is the order rule the inverse also needs, so they read
# the same values off the finished model point `y`. The map's gradient is
# therefore a reverse sweep over the runs that carries `ybar`, the running
# adjoint of `y`, and reads `y` itself, which it never writes:
#
#   * `ybar` starts as `g_y`, the seed of the model point;
#   * runs are visited last to first. A pair's output coordinate takes that
#     coordinate's remaining seed and zeroes it, which is what the reverse of
#     `y[idx] = …` does, so with a repeated index only its last writer is seeded;
#   * a run's own coordinates are read from `x`, with `gradient` as their shadow;
#     its frames are read from `y`, with `ybar` as its shadow. What a frame reads
#     of an earlier run adds to that run's seed; what it reads of a coordinate no
#     pair transforms is added to the gradient with the rest of `ybar` once every
#     run is done, as the reverse of `copy(x)`.
#
# The inverse reads every run's frames off the model point too, so its gradient
# takes one call per run with no state to restore.
#
# This replays the one-pass reverse sweep in its order, so the gradient agrees
# with it to rounding. Only the order in which a coordinate that several pairs
# read sums their contributions can differ, which the one-pass code leaves to
# the compiler.

#
# A Markov chain (`WarmupHMC.MarkovChain`) is differentiated in one `autodiff`
# call per chain: one reverse sweep over its states, whose tape holds a few
# numbers per state. Its accessors are evaluated once per chain. The map writes
# the chains before the pairs, so their reverse comes after the pairs' sweep and
# takes whatever `ybar` holds at their states, before the rest of `ybar` goes to
# the gradient. Their accessors read coordinates nothing transforms.

using WarmupHMC, Enzyme
using WarmupHMC: IndexedReparametrization, reparam, _reparam_framed, _run_start,
    _is_identity, _first_moved, _share_pattern, _patterned_frames,
    _chain_frames, _chain_step, _chain_unstep, _each_chain
using DifferentiationInterface: AutoEnzyme

# One pair's share of `_reparam_objective`: its log-Jacobian plus `seed` times
# the coordinate it writes, its accessors read off the model point `y`.
_pair_objective(value, idx, seed, x, y) = begin
    ljac, out = reparam(value, x[idx], y)
    ljac + seed * out
end

# A run's share: pairs `i:j` from one evaluation of their shared values, each
# seeded by its coordinate's entry of `seed`.
_run_objective(pairs, i, j, seed, x, y, pattern) = begin
    frames = _patterned_frames(last(pairs[i]), y, pattern)
    total = zero(eltype(x))
    for k in i:j
        idx, value = pairs[k]
        ljac, out = _reparam_framed(value, x[idx], frames)
        total += ljac + seed[idx] * out
    end
    total
end
# The same for a run that holds identity pairs, which it leaves out. Runs without
# them keep the objective above, with no branch per pair for the reverse pass.
_moved_run_objective(pairs, i, j, seed, x, y, pattern) = begin
    frames = _patterned_frames(last(pairs[_first_moved(pairs, i, j)]), y, pattern)
    total = zero(eltype(x))
    for k in i:j
        idx, value = pairs[k]
        _is_identity(value) && continue
        ljac, out = _reparam_framed(value, x[idx], frames)
        total += ljac + seed[idx] * out
    end
    total
end
_has_identity(pairs, i, j) = any(k -> _is_identity(last(pairs[k])), i:j)
# Which accessor positions of a run share a frame, as a type (see
# `WarmupHMC._frames_by`): a pattern of all zeros has one type per rule type, so
# a run whose positions share nothing dispatches statically.
_pattern(pairs, i) = (p = _share_pattern(last(pairs[i])); all(iszero, p) ? Val(map(zero, p)) : Val(p))

_autodiff_run(mode, objective, pairs, i, j, seed, x, gradient, y, ybar, pattern) =
    Enzyme.autodiff(mode, objective, Enzyme.Active, Enzyme.Const(pairs),
                    Enzyme.Const(i), Enzyme.Const(j), Enzyme.Const(seed),
                    Enzyme.Duplicated(x, gradient), Enzyme.Duplicated(y, ybar),
                    Enzyme.Const(pattern))

_reverse_mode(::AutoEnzyme{Nothing}) = Enzyme.Reverse
_reverse_mode(backend::AutoEnzyme{<:Enzyme.ReverseMode}) = backend.mode
_reverse_mode(::AutoEnzyme) = nothing

# Enzyme's own check of its concurrent-GC repair (`src/concurrent_gradients.jl`):
# the pass that carries it exists from dogfood `72d5d25`. The defect is in how
# Enzyme compiles a gradient, not in one mode, so every Enzyme backend asks.
WarmupHMC._unsafe_concurrent_gradients(::AutoEnzyme) =
    isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass) ? nothing :
    "This Enzyme ($(pkgversion(Enzyme)), loaded from $(pkgdir(Enzyme))) lacks the " *
    "concurrent-GC repair: `Enzyme.Compiler.RestoreSafepointEffectsPass` is not defined."

# Add into `gradient` the pairs' terms of the gradient in `x` of
# `ljac(x) + dot(ybar, y(x))`, where `y` is `y(x)`, read and never written. `ybar`
# is consumed at the pairs' coordinates, and what their frames read is added to it;
# the caller passes the rest on (the chains', then `copy(x)`'s). `sd` holds the
# seeds of a run while it is differentiated. One bookkeeping pass per run, and
# none at all for a run of one pair, keeps the sweep's cost that of the one-pass
# sweep it replays.
function _accumulate_pairs!(gradient, ybar, sd, y, mode, pairs, x)
    j = lastindex(pairs)
    while j >= firstindex(pairs)
        i = _run_start(pairs, j)
        if i == j
            idx, value = pairs[j]
            s = ybar[idx]
            ybar[idx] = 0
            if _is_identity(value)
                # an identity pair writes its own coordinate: its derivative is its seed
                gradient[idx] += s
            else
                Enzyme.autodiff(mode, _pair_objective, Enzyme.Active, Enzyme.Const(value),
                                Enzyme.Const(idx), Enzyme.Const(s),
                                Enzyme.Duplicated(x, gradient), Enzyme.Duplicated(y, ybar))
            end
        else
            moved = false
            for k in i:j
                idx, value = pairs[k]
                if _is_identity(value)
                    gradient[idx] += ybar[idx]
                else
                    moved = true
                    sd[idx] = ybar[idx]
                end
                ybar[idx] = 0
            end
            if moved
                objective = _has_identity(pairs, i, j) ? _moved_run_objective : _run_objective
                _autodiff_run(mode, objective, pairs, i, j, sd, x, gradient, y, ybar,
                              _pattern(pairs, i))
            end
        end
        j = i - 1
    end
    gradient
end

function WarmupHMC._reparam_gradient(backend::AutoEnzyme, ir::IndexedReparametrization,
                                     x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat}, g_y)
    mode = _reverse_mode(backend)
    mode === nothing && return invoke(WarmupHMC._reparam_gradient,
                                      Tuple{Any,Any,Any,Any,Any}, backend, ir, x, y, g_y)
    gradient = zero(x)
    ybar = copyto!(similar(x), g_y)
    _accumulate_pairs!(gradient, ybar, similar(x), y, mode, ir.pairs, x)
    _accumulate_chains!(gradient, ybar, mode, ir.chains, x)
    gradient .+= ybar
end

# --- Markov chains -----------------------------------------------------------------

# One chain's share of `_reparam_objective`: its log-Jacobian plus each state's
# seed times its model value.
_chain_objective(chain, seed, x) = begin
    frames = _chain_frames(chain, x)
    total = zero(eltype(x))
    previous = zero(eltype(x))
    for (t, idx) in enumerate(chain.states)
        log_step, yt = _chain_step(chain.phi[t], chain.c[t], frames, t, previous, x[idx])
        total += log_step + seed[idx] * yt
        previous = yt
    end
    total
end

# Accumulate into `gradient` the chains' terms of the gradient of
# `ljac(x) + dot(seed, y(x))`, consuming `seed` at their states.
_accumulate_chains!(gradient, seed, mode, chains, x) =
    (_each_chain(chain -> _accumulate_chain!(gradient, seed, mode, chain, x), chains); gradient)
function _accumulate_chain!(gradient, seed, mode, chain, x)
    if _is_identity(chain)
        for idx in chain.states
            gradient[idx] += seed[idx]
        end
    else
        Enzyme.autodiff(mode, _chain_objective, Enzyme.Active, Enzyme.Const(chain),
                        Enzyme.Const(seed), Enzyme.Duplicated(x, gradient))
    end
    for idx in chain.states
        seed[idx] = 0
    end
    nothing
end

# One chain's share of the inverse: its log-Jacobian plus each state's seed times
# its recovered source value. It reads only the model point `m`.
_chain_inverse_objective(chain, seed, m) = begin
    frames = _chain_frames(chain, m)
    states = chain.states
    total = zero(eltype(m))
    for (t, idx) in enumerate(states)
        previous = t == 1 ? zero(eltype(m)) : m[states[t - 1]]
        ljac, st = _chain_unstep(chain.phi[t], chain.c[t], frames, t, previous, m[idx])
        total += ljac + seed[idx] * st
    end
    total
end

# The inverse reads the chains' frames and previous states off the model point
# `m`, as the pairs' runs, so their reverse needs no order with the pairs': each
# takes its states' seeds from `ds`.
_accumulate_inverse_chains!(dm, ds, sd, mode, chains, m) =
    (_each_chain(chain -> _accumulate_inverse_chain!(dm, ds, sd, mode, chain, m), chains); dm)
function _accumulate_inverse_chain!(dm, ds, sd, mode, chain, m)
    if _is_identity(chain)
        for idx in chain.states
            dm[idx] += ds[idx]
            ds[idx] = 0
        end
    else
        for idx in chain.states
            sd[idx] = ds[idx]
            ds[idx] = 0
        end
        Enzyme.autodiff(mode, _chain_inverse_objective, Enzyme.Active,
                        Enzyme.Const(chain), Enzyme.Const(sd), Enzyme.Duplicated(m, dm))
    end
    nothing
end

# --- window-end halo transport ---------------------------------------------------
#
# `_transport_objective!` maps `x` forward through `new_ir` to the model point `m`
# and back through `old_ir`'s inverse to the old point `s`. The inverse reads
# every run's frames off `m`, so its derivative in `m` takes one call per run,
# seeded by `g_old` at the coordinates the run recovers. Visiting the runs last
# to first lets a repeated index seed only its last writer, as for the map. An
# identity pair recovers its target value, so its seed passes to `m` unchanged.
# The derivative in `m` then seeds the map's reverse sweep from `x`, which reads
# `m` as the map's model point.

# One inverse pair's share: its log-Jacobian plus `seed` times the coordinate it
# recovers.
_inverse_pair_objective(value, idx, seed, m) = begin
    ljac, s = reparam(WarmupHMC.inverse(value), m[idx], m)
    ljac + seed * s
end

# A run's share of the inverse: pairs `i:j` recovered from one evaluation of
# their shared values at `m`, each seeded by its coordinate's entry of `seed`.
_inverse_run_objective(pairs, i, j, seed, m, pattern) = begin
    frames = _patterned_frames(last(pairs[i]), m, pattern)
    total = zero(eltype(m))
    for k in i:j
        idx, value = pairs[k]
        ljac, s = _reparam_framed(WarmupHMC.inverse(value), m[idx], frames)
        total += ljac + seed[idx] * s
    end
    total
end
# The same for a run that holds identity pairs, which it leaves out.
_inverse_moved_run_objective(pairs, i, j, seed, m, pattern) = begin
    frames = _patterned_frames(last(pairs[_first_moved(pairs, i, j)]), m, pattern)
    total = zero(eltype(m))
    for k in i:j
        idx, value = pairs[k]
        _is_identity(value) && continue
        ljac, s = _reparam_framed(WarmupHMC.inverse(value), m[idx], frames)
        total += ljac + seed[idx] * s
    end
    total
end

_autodiff_inverse_run(mode, objective, pairs, i, j, sd, m, dm, pattern) =
    Enzyme.autodiff(mode, objective, Enzyme.Active, Enzyme.Const(pairs),
                    Enzyme.Const(i), Enzyme.Const(j), Enzyme.Const(sd),
                    Enzyme.Duplicated(m, dm), Enzyme.Const(pattern))

function _transport_gradient!(g_new, m, dm, ds, sd, mode, new_ir, old_ir, x, g_old)
    WarmupHMC._with_logabsdet_jacobian_into!(m, new_ir, x)
    fill!(dm, 0)
    copyto!(ds, g_old)
    pairs = old_ir.pairs
    j = lastindex(pairs)
    while j >= firstindex(pairs)
        i = _run_start(pairs, j)
        if i == j
            idx, value = pairs[j]
            s = ds[idx]
            ds[idx] = 0
            if _is_identity(value)
                # an identity pair's coordinate is its target value
                dm[idx] += s
            else
                Enzyme.autodiff(mode, _inverse_pair_objective, Enzyme.Active,
                                Enzyme.Const(value), Enzyme.Const(idx),
                                Enzyme.Const(s), Enzyme.Duplicated(m, dm))
            end
        else
            moved = false
            for k in i:j
                idx, value = pairs[k]
                if _is_identity(value)
                    dm[idx] += ds[idx]
                else
                    moved = true
                    sd[idx] = ds[idx]
                end
                ds[idx] = 0
            end
            if moved
                objective = _has_identity(pairs, i, j) ? _inverse_moved_run_objective :
                                                         _inverse_run_objective
                _autodiff_inverse_run(mode, objective, pairs, i, j, sd, m, dm,
                                      _pattern(pairs, i))
            end
        end
        j = i - 1
    end
    _accumulate_inverse_chains!(dm, ds, sd, mode, old_ir.chains, m)
    dm .+= ds
    fill!(g_new, 0)
    _accumulate_pairs!(g_new, dm, sd, m, mode, new_ir.pairs, x)
    _accumulate_chains!(g_new, dm, mode, new_ir.chains, x)
    g_new .+= dm
end

function WarmupHMC._transport_gradient_method(backend::AutoEnzyme,
                                              new_ir::IndexedReparametrization,
                                              old_ir::IndexedReparametrization,
                                              x::Vector{<:AbstractFloat}, g_old)
    mode = _reverse_mode(backend)
    mode === nothing && return invoke(WarmupHMC._transport_gradient_method,
                                      Tuple{Any,Any,Any,Any,Any},
                                      backend, new_ir, old_ir, x, g_old)
    m, dm, ds, sd = ntuple(_ -> similar(x), 4)
    (g_new, x, g_old) -> _transport_gradient!(g_new, m, dm, ds, sd, mode, new_ir, old_ir, x, g_old)
end

end
