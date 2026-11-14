# --- Gradients on concurrent chains --------------------------------------------
#
# Registered Enzyme releases compile gradients that are not memory-safe while
# another thread of the process allocates: a garbage collection that thread
# starts can free a value the gradient still uses, which crashes the process or,
# on Julia 1.12, returns a wrong gradient. Enzyme's dogfood line repairs it from
# `72d5d25` (enzyme-use, "Reverse gradients beside other threads (concurrent
# GC)"); the defect and its repair are Enzyme's (Enzyme snag
# `concurrent-gc-re-a24dcf7e`). Chains that WarmupHMC runs at the same time are
# such threads for one another. So every sampler that would run chains
# concurrently refuses, before any work, a density whose own gradients WarmupHMC
# takes with an Enzyme that lacks the repair: the map of a
# `ReparametrizedProblem` whose `ad_backend` is Enzyme's (snag
# `warmuphmc-concur-06de2cf4`). It does not route around the defect by running
# the chains one at a time: the remedy is the repaired Enzyme.
#
# A density's own gradient (the inner problem of a `ReparametrizedProblem`, or a
# plain target) is the caller's, and so is the check that its AD is safe on
# threads; WarmupHMC cannot see which AD produced it.

"""
    _unsafe_concurrent_gradients(backend) -> Union{Nothing,String}

`nothing` when the gradients WarmupHMC takes with `backend` stay sound while
other threads of the process allocate; otherwise why not, in a sentence.
`WarmupHMCEnzymeExt` answers for an Enzyme backend.
"""
_unsafe_concurrent_gradients(backend) = nothing

# The backend WarmupHMC differentiates `lpdf`'s reparametrization with; `nothing`
# for a density WarmupHMC takes no gradient of itself. An empty
# `IndexedReparametrization` maps nothing, and the reverse-mode Enzyme extension
# takes no gradient of it.
_own_gradient_backend(lpdf) = nothing
_own_gradient_backend(p::ReparametrizedProblem) =
    has_reparametrization(p.reparametrizer) ? p.ad_backend : nothing
_own_gradient_backend(p::WrappedLogDensityProblem) = _own_gradient_backend(parent(p))

# How many chains run at once: `n_chains` scheduled on `n_workers` tasks of the
# default thread pool. Julia's `@threads` and `@spawn` use that pool only.
_concurrent_chains(n_chains, n_workers) = max(1, min(n_chains, n_workers, Threads.nthreads()))

"""
    _refuse_unsafe_concurrent_gradients(caller, lpdfs, n_concurrent)

Throw an `ArgumentError` when `n_concurrent > 1` chains would run at once and
WarmupHMC would take one of `lpdfs`' reparametrization gradients with an AD
backend that is not memory-safe beside other threads.
"""
function _refuse_unsafe_concurrent_gradients(caller, lpdfs, n_concurrent)
    n_concurrent > 1 || return nothing
    for lpdf in lpdfs
        reason = _unsafe_concurrent_gradients(_own_gradient_backend(lpdf))
        isnothing(reason) && continue
        throw(ArgumentError(
            "$caller would run $n_concurrent chains at once (Julia has $(Threads.nthreads()) " *
            "threads), and each takes Enzyme gradients of its ReparametrizedProblem's " *
            "reparametrization. $reason A garbage collection that one chain's thread " *
            "starts can then free memory another chain's gradient still uses: the " *
            "process crashes, or a gradient is silently wrong. Use Enzyme from the " *
            "`dogfood` branch of https://github.com/nsiccha/Enzyme.jl (72d5d25 or " *
            "later), which carries the repair."))
    end
    nothing
end
