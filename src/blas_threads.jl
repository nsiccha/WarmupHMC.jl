# ── BLAS threads for the duration of a sampler call ─────────────────────────
#
# Every leapfrog step takes several `dot`s of posterior-length vectors:
# DynamicHMC's kinetic energy and U-turn check, and the reflections of the
# `:adaptive` metric. OpenBLAS runs `dot` on all of its threads once the vectors
# are longer than 10,000 entries, and its workers then busy-wait for the next
# call instead of sleeping. So on a large posterior the default pool (one thread
# per core) spins through every gradient evaluation. Under a CPU quota, or with
# chains on Julia threads, those spinning workers take the CPU the gradient
# needs: measured on a 1-CPU quota at d = 25,742, a gradient that costs 12 ms
# alone cost 96 ms inside the sampler with 16 BLAS threads, and 12.5 ms with one
# (snag `openblas-threads-2f1bf9e9`). Single-threaded BLAS is also what makes a
# fixed seed reproduce its draws: the metric fit's multithreaded reductions are
# not deterministic.
#
# Hence every sampler entry point runs under `blas_threads` BLAS threads, one by
# default, and gives back the count it found. The setting is process-global, so
# the pin is shared: nested calls (completion → adaptive per chain) and
# concurrent calls in one process run under the FIRST active pin, and the last
# call to leave restores the count that pin found. `blas_threads=nothing` leaves
# BLAS untouched.

const _DEFAULT_BLAS_THREADS = 1
const _BLAS_PIN_LOCK = ReentrantLock()
const _BLAS_PIN_DEPTH = Ref(0)
const _BLAS_PIN_THREADS = Ref(0)
const _BLAS_PIN_SAVED = Ref(0)

"""
    _with_blas_threads(f, blas_threads)

Run `f()` with `blas_threads` BLAS threads and restore the caller's count
afterwards, also when `f` throws. `nothing` runs `f()` with BLAS untouched.
While another call holds a pin with a different count, `f` runs under that
count, with a warning.
"""
_with_blas_threads(f, ::Nothing) = f()
_with_blas_threads(f, n::Integer) = begin
    (n isa Bool || n < 1) && throw(ArgumentError(
        "`blas_threads` must be a positive integer or `nothing`; got $(repr(n))."))
    held = lock(_BLAS_PIN_LOCK) do
        if _BLAS_PIN_DEPTH[] == 0
            _BLAS_PIN_SAVED[] = BLAS.get_num_threads()
            _BLAS_PIN_THREADS[] = n
            _BLAS_PIN_SAVED[] == n || BLAS.set_num_threads(n)
        end
        _BLAS_PIN_DEPTH[] += 1
        _BLAS_PIN_THREADS[]
    end
    held == n || @warn "`blas_threads=$n` was requested while another WarmupHMC call in this process " *
        "holds BLAS at $held thread(s); this call runs with $held. The BLAS thread count is " *
        "process-global, so concurrent sampler calls share one setting." maxlog = 1
    try
        f()
    finally
        lock(_BLAS_PIN_LOCK) do
            _BLAS_PIN_DEPTH[] -= 1
            if _BLAS_PIN_DEPTH[] == 0 && BLAS.get_num_threads() != _BLAS_PIN_SAVED[]
                BLAS.set_num_threads(_BLAS_PIN_SAVED[])
            end
        end
    end
end
_with_blas_threads(f, n) = throw(ArgumentError(
    "`blas_threads` must be a positive integer or `nothing`; got $(repr(n))."))
