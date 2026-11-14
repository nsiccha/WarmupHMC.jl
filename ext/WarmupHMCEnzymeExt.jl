module WarmupHMCEnzymeExt

# Reverse-mode Enzyme gradients of an `IndexedReparametrization`, pair by pair.
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
# The per-pair pass replays what the one-pass reverse sweep does, in its order,
# so the gradient agrees with it to rounding. Only the order in which a coordinate
# that several pairs read sums their contributions can differ, which the one-pass
# code leaves to the compiler:
#
#   * pairs are visited last to first, accumulating into one shadow of `x`;
#   * a pair's output coordinate takes that coordinate's remaining seed and
#     zeroes it, which is what the reverse of `y[idx] = …` does — so with a
#     repeated index only its last writer is seeded;
#   * the reverse of `copy(x)` comes last: whatever seed is left, on the
#     coordinates no pair wrote, is added to the gradient.
#
# Pairs that share accessor work (a run, `WarmupHMC._run_end`) are differentiated
# together, one `autodiff` per run, so their shared location or log-scale is
# evaluated and differentiated once rather than once per pair (snag
# `use-the-public-i-0660224b`). A run's indices increase, so each of its pairs
# takes its own coordinate's seed. A run of one pair is the pair-by-pair call.
# A pair whose source and target centerings are equal is the identity on its
# coordinate: it takes no call, its derivative is its seed, and a run of such
# pairs evaluates no accessor (snag `centering-evalua-54d5e611`).

using WarmupHMC, Enzyme
using WarmupHMC: IndexedReparametrization, reparam, _reparam_framed, _run_start,
    _is_identity, _any_moved, _first_moved, _share_pattern, _patterned_frames
using DifferentiationInterface: AutoEnzyme

# One pair's share of `_reparam_objective`: its log-Jacobian plus `seed` times
# the coordinate it writes.
_pair_objective(value, idx, seed, x) = begin
    ljac, y = reparam(value, x[idx], x)
    ljac + seed * y
end

# A run's share: pairs `i:j` from one evaluation of their shared values, each
# seeded by its coordinate's entry of `seed`.
_run_objective(pairs, i, j, seed, x, pattern) = begin
    frames = _patterned_frames(last(pairs[i]), x, pattern)
    total = zero(eltype(x))
    for k in i:j
        idx, value = pairs[k]
        ljac, y = _reparam_framed(value, x[idx], frames)
        total += ljac + seed[idx] * y
    end
    total
end
# The same for a run that holds identity pairs, which it leaves out. Runs without
# them keep the objective above, with no branch per pair for the reverse pass.
_moved_run_objective(pairs, i, j, seed, x, pattern) = begin
    frames = _patterned_frames(last(pairs[_first_moved(pairs, i, j)]), x, pattern)
    total = zero(eltype(x))
    for k in i:j
        idx, value = pairs[k]
        _is_identity(value) && continue
        ljac, y = _reparam_framed(value, x[idx], frames)
        total += ljac + seed[idx] * y
    end
    total
end
_has_identity(pairs, i, j) = any(k -> _is_identity(last(pairs[k])), i:j)
# Which accessor positions of a run share a frame, as a type (see
# `WarmupHMC._frames_by`): a pattern of all zeros has one type per rule type, so
# a run whose positions share nothing dispatches statically.
_pattern(pairs, i) = (p = _share_pattern(last(pairs[i])); all(iszero, p) ? Val(map(zero, p)) : Val(p))

_autodiff_run(mode, objective, pairs, i, j, seed, x, gradient, pattern) =
    Enzyme.autodiff(mode, objective, Enzyme.Active, Enzyme.Const(pairs),
                    Enzyme.Const(i), Enzyme.Const(j), Enzyme.Const(seed),
                    Enzyme.Duplicated(x, gradient), Enzyme.Const(pattern))

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

# Accumulate into `gradient` the pairs' terms of the gradient of
# `ljac(x) + dot(seed, y(x))`, consuming `seed` at the coordinates the pairs write.
function _accumulate_pairs!(gradient, seed, mode, pairs, x)
    j = lastindex(pairs)
    while j >= firstindex(pairs)
        i = _run_start(pairs, j)
        if !_any_moved(pairs, i, j)
        elseif i == j
            idx, value = pairs[j]
            s = seed[idx]
            seed[idx] = 0
            Enzyme.autodiff(mode, _pair_objective, Enzyme.Active, Enzyme.Const(value),
                            Enzyme.Const(idx), Enzyme.Const(s), Enzyme.Duplicated(x, gradient))
        else
            objective = _has_identity(pairs, i, j) ? _moved_run_objective : _run_objective
            _autodiff_run(mode, objective, pairs, i, j, seed, x, gradient, _pattern(pairs, i))
        end
        # An identity pair writes its own coordinate: its derivative is its seed.
        for k in i:j
            idx, value = pairs[k]
            _is_identity(value) && (gradient[idx] += seed[idx])
            seed[idx] = 0
        end
        j = i - 1
    end
    gradient
end

function WarmupHMC._reparam_gradient(backend::AutoEnzyme, ir::IndexedReparametrization,
                                     x::Vector{<:AbstractFloat}, g_y)
    mode = _reverse_mode(backend)
    mode === nothing && return invoke(WarmupHMC._reparam_gradient,
                                      Tuple{Any,Any,Any,Any}, backend, ir, x, g_y)
    gradient = zero(x)
    seed = copyto!(similar(x), g_y)
    _accumulate_pairs!(gradient, seed, mode, ir.pairs, x)
    gradient .+= seed
end

# --- window-end halo transport ---------------------------------------------------
#
# `_transport_objective!` maps `x` forward through `new_ir` to the model point `m`
# and back through `old_ir`'s inverse to the old point `s`. The inverse runs its
# pairs in order on one vector, so an accessor reads the coordinates earlier
# pairs have recovered and the target values of the rest. Replaying it backwards
# restores that state: before differentiating pair `i`, its coordinate is set
# back to its target value. That needs each index to occur once. An identity pair
# recovers its target value, so its seed passes to `m` unchanged.

# One inverse pair's share: its log-Jacobian plus `seed` times the coordinate it
# recovers.
_inverse_pair_objective(value, idx, seed, target, source) = begin
    ljac, s = reparam(WarmupHMC.inverse(value), target[idx], source)
    ljac + seed * s
end

# A run's share of the inverse: pairs `i:j` recovered from one evaluation of their
# shared values at the run's starting `source`, each seeded by its coordinate's
# entry of `seed`.
_inverse_run_objective(pairs, i, j, seed, target, source, pattern) = begin
    frames = _patterned_frames(last(pairs[i]), source, pattern)
    total = zero(eltype(source))
    for k in i:j
        idx, value = pairs[k]
        ljac, s = _reparam_framed(WarmupHMC.inverse(value), target[idx], frames)
        total += ljac + seed[idx] * s
    end
    total
end
# The same for a run that holds identity pairs, which it leaves out.
_inverse_moved_run_objective(pairs, i, j, seed, target, source, pattern) = begin
    frames = _patterned_frames(last(pairs[_first_moved(pairs, i, j)]), source, pattern)
    total = zero(eltype(source))
    for k in i:j
        idx, value = pairs[k]
        _is_identity(value) && continue
        ljac, s = _reparam_framed(WarmupHMC.inverse(value), target[idx], frames)
        total += ljac + seed[idx] * s
    end
    total
end

_autodiff_inverse_run(mode, objective, pairs, i, j, sd, m, dm, s, ds, pattern) =
    Enzyme.autodiff(mode, objective, Enzyme.Active, Enzyme.Const(pairs),
                    Enzyme.Const(i), Enzyme.Const(j), Enzyme.Const(sd),
                    Enzyme.Duplicated(m, dm), Enzyme.Duplicated(s, ds), Enzyme.Const(pattern))

function _transport_gradient!(g_new, m, s, dm, ds, sd, mode, new_ir, old_ir, x, g_old)
    WarmupHMC._with_logabsdet_jacobian_into!(m, new_ir, x)
    WarmupHMC._inverse_with_logabsdet_jacobian!(s, old_ir, m)
    fill!(dm, 0)
    copyto!(ds, g_old)
    pairs = old_ir.pairs
    j = lastindex(pairs)
    while j >= firstindex(pairs)
        i = _run_start(pairs, j)
        if !_any_moved(pairs, i, j)
        elseif i == j
            idx, value = pairs[j]
            s[idx] = m[idx]
            seed = ds[idx]
            ds[idx] = 0
            Enzyme.autodiff(mode, _inverse_pair_objective, Enzyme.Active, Enzyme.Const(value),
                            Enzyme.Const(idx), Enzyme.Const(seed),
                            Enzyme.Duplicated(m, dm), Enzyme.Duplicated(s, ds))
        else
            # Back to the state where the run starts; its seeds move out of `ds`,
            # which accumulates the run's own derivative in `source`.
            for k in i:j
                idx, value = pairs[k]
                s[idx] = m[idx]
                _is_identity(value) && continue
                sd[idx] = ds[idx]
                ds[idx] = 0
            end
            objective = _has_identity(pairs, i, j) ? _inverse_moved_run_objective :
                                                     _inverse_run_objective
            _autodiff_inverse_run(mode, objective, pairs, i, j, sd, m, dm, s, ds,
                                  _pattern(pairs, i))
        end
        # An identity pair's coordinate is its target value.
        for k in i:j
            idx, value = pairs[k]
            _is_identity(value) || continue
            dm[idx] += ds[idx]
            ds[idx] = 0
        end
        j = i - 1
    end
    dm .+= ds
    fill!(g_new, 0)
    _accumulate_pairs!(g_new, dm, mode, new_ir.pairs, x)
    g_new .+= dm
end

function WarmupHMC._transport_gradient_method(backend::AutoEnzyme,
                                              new_ir::IndexedReparametrization,
                                              old_ir::IndexedReparametrization,
                                              x::Vector{<:AbstractFloat}, g_old)
    mode = _reverse_mode(backend)
    if mode === nothing || !allunique(first(pair) for pair in old_ir.pairs)
        return invoke(WarmupHMC._transport_gradient_method, Tuple{Any,Any,Any,Any,Any},
                      backend, new_ir, old_ir, x, g_old)
    end
    m, s, dm, ds, sd = ntuple(_ -> similar(x), 5)
    (g_new, x, g_old) -> _transport_gradient!(g_new, m, s, dm, ds, sd, mode, new_ir, old_ir, x, g_old)
end

end
