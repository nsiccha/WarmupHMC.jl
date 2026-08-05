# ============================================================================
# Interruptible, resumable, FIXED-KERNEL streaming NUTS sampler.
#
# This is the "append-only production-draw phase" the adaptive sampler is NOT:
# pure sampling, NO warm-up and NO adaptation. Given an initial point and a fixed
# kernel (a mass-matrix SCALE + a step size + a max tree depth), it runs NUTS
# transitions and streams each draw, column by column, into an mmappable
# `Float64` file that is zeroed at creation. A process killed mid-sample or
# mid-write resumes from the immediately preceding state on the next call.
#
# WHAT IS PERSISTED, AND WHY IT IS MINIMAL.
#   * The DRAWS stream to the samples file already — so the ring does NOT store
#     the current position again; it is read back from the file on resume. The
#     ring holds only `(rng, n_written, dimension)`: the sampler's RNG (NOT in
#     the samples file — it is the transition's internal randomness), the count
#     of safely-written draws (the "last safe iteration"), and the column height.
#   * The kernel HYPERPARAMETERS (metric, step size, tree depth, draw count) are
#     never written to disk: they are re-supplied on every call, or read out of a
#     WarmupHMC checkpoint payload. This mirrors the adaptive sampler's own
#     "config comes from the resuming call, not the payload" contract.
#
# THE FILE IS THE SOURCE (WORKING) FRAME. Because resume reads the position back
# from the samples file, the file stores the sampler's raw source-frame draws —
# the chain state, exactly what `restore_state` needs. Model-frame (constrained)
# draws are produced on READ, via [`open_stream`](@ref)`(path, lpdf; model=true)`
# or `WarmupHMC.back_transform`, precisely as a checkpoint payload's raw draws
# are. For a plain log density the two frames coincide, so the file is already
# the posterior draws.
#
# THE FILES, for an output path `P` (each a plain mmappable array, zeroed at
# creation; the three per-draw arrays grow/resume in lockstep):
#   * `P`               — the draws: a `dimension × n_draws` column-major
#                    `Float64` matrix, memory-mappable as-is. Draws are COLUMNS.
#   * `P * ".rng_states"` — per-draw RNG state: a `5 × n_draws` `UInt64` matrix,
#                    column k the full Xoshiro state (s0..s4) AFTER draw k.
#   * `P * ".divergences"`— per-draw divergence flag: a length-`n_draws` `Int8`
#                    vector, 1 if draw k diverged else 0 (their sum = divergences).
#   * `P * ".ring"`— the safe ring: two fixed-size slots written alternately,
#                    each `[seq | len | crc32c | serialize(rng, n_written, dim)]`.
#                    A torn write hits only one slot; the other still holds the
#                    previous complete state, and the CRC tells them apart.
#
# CRASH-SAFETY ORDERING, per draw k (0-based) written to column k+1 (1-based):
#   1. `sample_tree` → next position, mutating `rng`.
#   2. Store the column into the samples mmap AND the per-draw sidecars (divergent
#      flag + full post-draw RNG state). A store lands in the page cache, so it
#      survives a PROCESS crash without an explicit sync.
#   3. (`durable=true` only) `Mmap.sync!` samples + both sidecars, so they are on
#      DISK before the ring can point at them — machine-crash safety.
#   4. Commit `(rng, n_written=k+1, dim)` to the ALTERNATE ring slot.
# The ring's commit `n` happens strictly AFTER column `n` is written, so
# "the ring says `n`" implies columns `1..n` are complete: a crash between (2)
# and (4) leaves the ring at `n-1`, and re-running the fixed, deterministic
# transition from `(rng, position)` reproduces the exact column that was (maybe
# partially) written, then overwrites it. The ring's first commit is DEFERRED
# until after draw 1, so a crash inside the very first transition is simply not
# resumable (zero draws lost) and restarts from the caller's seed.
#
# The ring persists rng + index only; the running divergence count is NOT in it,
# so the returned `n_divergent` counts the draws of THE CURRENT call.
# ============================================================================

# ------------------------------------------------------------------- kernel

# The WarmupHMC-native kinetic energy from a mass-matrix SCALE `L` — identical to
# the construction the adaptive/cooperative/clustered samplers use, so a scale
# `L` learned by warm-up (`scale_options[active_transformation]`) plugs straight
# in. `L * L'` is the momentum covariance; a diagonal `L`'s entries are
# per-coordinate standard deviations (cf. `_initial_diagonal_scale`).
_stream_kinetic_energy(L) =
    DynamicHMC.GaussianKineticEnergy(MatrixFactorization(L, L'), MatrixInverse(L'))

# Normalize a caller-supplied `metric` to a DynamicHMC kinetic energy. Accepted:
#   * a `DynamicHMC.GaussianKineticEnergy` — used as-is (full control);
#   * a `Diagonal`/`AbstractMatrix` (or a WarmupHMC matrix expression) — the
#     scale `L`;
#   * an `AbstractVector` — per-coordinate scales (std devs), i.e. `Diagonal(v)`.
_stream_energy(m::DynamicHMC.GaussianKineticEnergy) = m
_stream_energy(m::AbstractMatrix) = _stream_kinetic_energy(m)
_stream_energy(m::AbstractVector) = _stream_kinetic_energy(Diagonal(collect(float.(m))))

# Extract the fixed kernel from a WarmupHMC checkpoint payload AND restore its
# learned reparametrization centerings onto `lpdf`, so `lpdf`'s frame matches the
# checkpoint's. Returns `(kinetic_energy, stepsize)`. Shared by every
# checkpoint-seeded entry point; called once per chain (on that chain's own lpdf
# copy) in the multi-chain forms, which is why it takes and mutates `lpdf`.
_checkpoint_kernel!(payload::NamedTuple, lpdf) = begin
    check_checkpoint_compatible(payload, :adaptive, (:adaptive, :cooperative, :clustered))
    dimension = LogDensityProblems.dimension(lpdf)
    dimension == payload.dimension || throw(DimensionMismatch(
        "checkpoint holds a $(payload.dimension)-dimensional problem but the " *
        "supplied lpdf has dimension $dimension."))
    restore_reparam_sources!(lpdf, get(payload, :reparam_sources, Pair[]))
    (_stream_kinetic_energy(payload.scale_options[payload.active_transformation]), float(payload.stepsize))
end

# N independent, reproducible RNGs derived from one base RNG (a checkpoint's
# `rng`): draw N seeds from a copy of the base and reseed same-type copies with
# them. Used as the DEFAULT when a multi-chain caller passes no explicit `rngs`;
# distinct seeds are what keep the chains from sharing a random stream.
_derive_chain_rngs(base, n) = begin
    gen = copy(base)
    [(r = copy(base); Random.seed!(r, rand(gen, UInt64)); r) for _ in 1:n]
end

_stream_lpdfs(lpdf, n_chains) =
    lpdf isa AbstractVector && length(lpdf) == n_chains ? collect(lpdf) :
    [deepcopy(lpdf) for _ in 1:n_chains]   # a stateful wrapper must not be shared across chains

# ---------------------------------------------------------------- safe ring

# One slot's fixed prefix: seq (UInt64) + payload length (UInt64) + crc32c
# (UInt32). The CRC covers those 16 header bytes AND the payload, so a torn write
# anywhere in the slot fails validation.
const _RING_HEADER = 20

# Handle for the ACTIVE ring writer. `next_slot`/`next_seq` say where the next
# commit goes; a commit always targets the slot NOT holding the freshest state,
# so the freshest complete state is never the one being overwritten.
mutable struct _StreamRing
    io::IOStream
    slot_cap::Int
    next_slot::Int      # 0 or 1
    next_seq::UInt64
end

_ring_path(path) = string(path, ".ring")

_serialize_ring(rng, n_written, dimension) = begin
    buf = IOBuffer()
    serialize(buf, (rng, n_written, dimension))
    take!(buf)
end

# Build a slot's bytes: [seq::UInt64][len::UInt64][crc::UInt32][payload].
_ring_slot_bytes(seq::UInt64, payload::Vector{UInt8}) = begin
    hdr = IOBuffer()
    write(hdr, seq)
    write(hdr, UInt64(length(payload)))
    hdr16 = take!(hdr)                                      # 16 bytes: seq + len
    crc = CRC32c.crc32c(payload, CRC32c.crc32c(hdr16))     # over hdr16 ++ payload
    out = IOBuffer()
    write(out, hdr16)
    write(out, crc)
    write(out, payload)
    take!(out)
end

_ring_commit!(ring::_StreamRing, rng, n_written, dimension) = begin
    payload = _serialize_ring(rng, n_written, dimension)
    _RING_HEADER + length(payload) <= ring.slot_cap || error(
        "safe-ring slot overflow: state serialized to $(length(payload)) bytes, " *
        "slot capacity is $(ring.slot_cap - _RING_HEADER). This should not happen " *
        "within a run (rng type and dimension are fixed).")
    bytes = _ring_slot_bytes(ring.next_seq, payload)
    seek(ring.io, ring.next_slot * ring.slot_cap)
    write(ring.io, bytes)
    flush(ring.io)                                          # to the page cache
    ring.next_slot = 1 - ring.next_slot
    ring.next_seq += one(UInt64)
    nothing
end

# Create the ring file (zeroed) and size its slots to the state plus a margin.
# NO slot is written yet — the first commit is deferred until after draw 1.
_ring_create(ring_path, seed_rng, dimension) = begin
    payload = _serialize_ring(seed_rng, 0, dimension)
    slot_cap = _RING_HEADER + length(payload) + 512        # margin; constant/run
    io = open(ring_path, "w+")
    truncate(io, 0)
    truncate(io, 2 * slot_cap)                             # two zeroed slots
    _StreamRing(io, slot_cap, 0, UInt64(1))
end

# Read one slot; return `(seq, rng, n_written, dimension)` or `nothing` (unwritten
# or torn — the CRC rejects a zeroed slot, since crc32c of a zeroed header is not
# the stored 0).
_ring_read_slot(io, offset, slot_cap) = begin
    slot_cap >= _RING_HEADER || return nothing
    seek(io, offset)
    seq = read(io, UInt64)
    len = read(io, UInt64)
    crc = read(io, UInt32)
    len <= UInt64(max(0, slot_cap - _RING_HEADER)) || return nothing
    payload = read(io, Int(len))
    length(payload) == Int(len) || return nothing
    hdr = IOBuffer(); write(hdr, seq); write(hdr, len); hdr16 = take!(hdr)
    CRC32c.crc32c(payload, CRC32c.crc32c(hdr16)) == crc || return nothing
    rng, n_written, dimension = deserialize(IOBuffer(payload))
    (seq, rng, n_written, dimension)
end

# Read the freshest valid slot: `(seq, rng, n_written, dimension, slot)` or
# `nothing` if neither slot is valid.
_ring_read(ring_path) = begin
    isfile(ring_path) || return nothing
    fsz = filesize(ring_path)
    (fsz > 0 && iseven(fsz)) || return nothing
    slot_cap = fsz ÷ 2
    open(ring_path, "r") do io
        best = nothing
        for slot in 0:1
            r = _ring_read_slot(io, slot * slot_cap, slot_cap)
            isnothing(r) && continue
            seq = r[1]
            (isnothing(best) || seq > best[1]) && (best = (seq, r[2], r[3], r[4], slot))
        end
        best
    end
end

_is_resumable(samples_path, ring_path) =
    isfile(samples_path) && !isnothing(_ring_read(ring_path))

# --------------------------------------------------- mmapped stream files

# The three per-draw mmapped sidecars of the samples file at `P`, one array each
# (each its own file, all zeroed at creation, all resumed/grown in lockstep):
#   * `P`               — the draws           `Float64`  `dimension × n_draws`
#   * `P * ".rng_states"` — per-draw RNG state `UInt64`   `5 × n_draws`   (Xoshiro)
#   * `P * ".divergences"`— per-draw divergent  `Int8`     `n_draws`       (0/1 flag)
_rng_states_path(path) = string(path, ".rng_states")
_divergences_path(path) = string(path, ".divergences")

# The FULL Xoshiro state: xoshiro256 words s0..s3 PLUS the internal splitmix
# word s4 (Julia 1.10's `Xoshiro` carries all five). Storing all five makes the
# per-draw RNG exactly reconstructable via `Xoshiro(s0, s1, s2, s3, s4)`.
const _XOSHIRO_WORDS = 5
_xoshiro_words(rng::Random.Xoshiro) = (rng.s0, rng.s1, rng.s2, rng.s3, rng.s4)
_xoshiro_words(rng) = throw(ArgumentError(
    "stream_mcmc persists the per-draw RNG state and currently supports only " *
    "`Random.Xoshiro` (you passed a $(typeof(rng))). Seed the sampler with a Xoshiro."))

# Open/create an mmappable array of element type `T` and shape `dims`. `fresh`
# (or a not-yet-existing file) truncates to zero then grows to `dims` (a grown
# region reads as zeros — "zeroed at creation"). A resume grows the file only if
# the new shape needs more room, never shrinks it. A resume over a run predating
# a given sidecar creates it zeroed: the already-drawn columns cannot be
# reconstructed (those draws are past), so they read as zeros; new columns are
# recorded truthfully.
_open_mmap(path, ::Type{T}, dims::Dims; fresh::Bool) where {T} = begin
    nbytes = prod(dims) * sizeof(T)
    make = fresh || !isfile(path)
    io = open(path, make ? "w+" : "r+")
    if make
        truncate(io, 0)
        truncate(io, nbytes)
    elseif filesize(path) < nbytes
        truncate(io, nbytes)
    end
    arr = Mmap.mmap(io, Array{T,length(dims)}, dims; shared=true)
    close(io)                                              # the mapping persists
    arr
end

# ------------------------------------------------------------- diagnostics

# Bulk-ESS per coordinate over the first `k` SOURCE-frame draws, sorted ascending
# (so element 1 is the worst-mixing coordinate). Same call the adaptive sampler
# monitors with — `ess` wants (draws, chains, params), and our draws are stored
# (params, draws), so transpose into a dense (k, 1, dimension) array. Computed on
# the working frame, exactly like the adaptive sampler's window ESS.
_stream_ess(samples, k, dimension) =
    sort!(MCMCDiagnosticTools.ess(reshape(collect(transpose(@view samples[:, 1:k])), (k, 1, dimension))))

# ------------------------------------------------------------- the driver

# The inner loop, shared by fresh runs and resumes. Mutates `samples` and the
# `ring` in place; consumes `rng`. Returns `(pg, rng, n_divergent, n_written)`.
#
# Progress (only when `progress !== nothing`): the counter is driven by the
# `@progress … for` macro, so it advances by one on EVERY draw for free — there
# is NOTHING throttled. Each draw also merges the NON-FIXED labels onto that same
# node — divergences, this session's draw rate, mean leapfrog steps per draw, and
# the ESS (the metric and step size are fixed, so they are deliberately NOT
# shown). All four are shown from the first draw; `ess` reads `pending...` until
# there are enough draws to compute it, then its VALUE. The label merge is O(1)
# (a lock + field set); only the ESS itself is O(k log k), so it is recomputed on
# EXPONENTIAL windows (each recompute at twice the previous draw count) while the
# last computed value stays on the label in between.
_stream_loop!(rng, lpdf, pg, kinetic_energy, stepsize;
        n_draws, max_tree_depth, ring, samples, rng_states, divergences,
        dimension, n_written, durable, progress, description, start_time) = begin
    algorithm = DynamicHMC.NUTS(; max_depth=max_tree_depth)
    hamiltonian = DynamicHMC.Hamiltonian(kinetic_energy, lpdf)
    steps_per_draw = OnlineStatsBase.Mean()
    total_steps = 0                                        # Σ leapfrog steps = Σ gradient evals
    ess_next = 16                                          # first ESS window
    ess_label = "pending..."                              # until the first window
    n_divergent = 0
    show = progress !== nothing                            # no ESS/label cost when off
    # `@progress <parent> "<desc>" for` hangs a determinate counter (labeled with
    # the runtime `description`) under `progress` and increments it every
    # iteration; inside the body `__progress__` IS that counter node, so the
    # label merge lands on the advancing bar. On resume it counts this session's
    # draws (n_written already on disk), while the labels report absolute counts.
    @progress progress "$description" for k in (n_written + 1):n_draws
        pg, stats = DynamicHMC.sample_tree(rng, algorithm, hamiltonian, pg, stepsize)
        @views samples[:, k] .= pg.q                      # raw SOURCE-frame draw
        diverged = DynamicHMC.is_divergent(stats.termination)
        divergences[k] = diverged ? Int8(1) : Int8(0)     # per-draw divergent flag
        @views rng_states[:, k] .= _xoshiro_words(rng)    # per-draw RNG state (post-draw)
        diverged && (n_divergent += 1)
        # Columns are in the page cache. For machine-crash durability, force them
        # to DISK before the ring is allowed to point at them.
        durable && (Mmap.sync!(samples); Mmap.sync!(rng_states); Mmap.sync!(divergences))
        _ring_commit!(ring, rng, k, dimension)
        if show
            # Each leapfrog step evaluates the gradient once, so Σ steps = Σ
            # gradient evaluations (DynamicHMC `stats.steps` = leapfrog steps).
            OnlineStatsBase.fit!(steps_per_draw, stats.steps)
            total_steps += stats.steps
            # ESS is the one O(k log k) update, so recompute it only on doubling
            # windows (and once at the end); the value persists between.
            if (k >= ess_next || k == n_draws) && k > 10
                ess_label = short_string(_stream_ess(samples, k, dimension)) * " from $k draws"
                ess_next = 2k
            end
            # One merge per draw carrying ALL non-fixed labels, so none is ever
            # blank or wiped by another. (`__progress__` = the counter node.)
            elapsed = time_ns() - start_time
            update_progress!(__progress__, nothing;
                ess = ess_label,
                divergent = UncertainFrequency(n_divergent, k - n_written),
                draws = Speed(k - n_written, elapsed),
                gradient_evals = Speed(total_steps, elapsed),   # leapfrog steps / s
                steps_per_draw = mean(steps_per_draw),
            )
        end
    end
    (pg, rng, n_divergent, n_draws)                       # n_draws columns now written
end

_stream_result(samples_path, samples, rng_states, divergences,
        dimension, n_written, n_divergent, ess, rng, position) = (;
    path = samples_path,
    draws = @view(samples[:, 1:n_written]),   # dimension × n_written SOURCE-frame draws (COLUMNS)
    samples,                                   # the live mapping — keeps `draws` valid
    n_drawn = n_written,
    n_divergent,                               # divergences of THIS call (not persisted)
    divergences = @view(divergences[1:n_written]),  # per-draw 0/1 flag, ALL draws on disk
    rng_states = @view(rng_states[:, 1:n_written]), # per-draw Xoshiro state (5 × n_written), post-draw
    ess,                                       # per-coordinate bulk-ESS, sorted ascending (NaN if <=10 draws)
    dimension,
    rng,                                       # rng state after the last draw
    position,                                  # source-frame chain position (resume point)
)

# The one place fresh and resume converge. `seed_rng`/`seed_position` are used
# only for a fresh run; a resume reads `(rng, n_written, dim)` from the ring and
# the last position from the samples file, ignoring them.
# `kinetic_energy`/`stepsize` are always caller-provided (from explicit
# hyperparameters or a checkpoint payload) — never read from disk.
_stream_impl(lpdf; samples_path, ring_path, resumable,
        seed_rng, seed_position, kinetic_energy, stepsize,
        n_draws, max_tree_depth, durable, progress, description) = begin
    dimension = LogDensityProblems.dimension(lpdf)
    if resumable
        seq, rng, n_written, dim_disk, last_slot = _ring_read(ring_path)
        dim_disk == dimension || throw(DimensionMismatch(
            "the safe ring at $(repr(ring_path)) describes a $dim_disk-dimensional " *
            "run but the supplied lpdf has dimension $dimension."))
        n_written >= 1 || error(
            "resumable ring at $(repr(ring_path)) reports n_written=$n_written; a " *
            "committed ring always has at least one draw. The file is corrupt.")
        eff_n_draws = max(n_draws, n_written)
        fresh = false
        # Resume position is the last safe draw, read BACK from the file (never
        # re-stored in the ring). Re-evaluate under this lpdf: a serialized
        # gradient is never trusted, and one evaluation is negligible.
        samples = _open_mmap(samples_path, Float64, (dimension, eff_n_draws); fresh)
        pg = DynamicHMC.evaluate_ℓ(lpdf, collect(@view samples[:, n_written]); strict=true)
        ring = _StreamRing(open(ring_path, "r+"), filesize(ring_path) ÷ 2, 1 - last_slot, seq + one(UInt64))
    else
        rng = seed_rng
        q0 = seed_position isa DynamicHMC.EvaluatedLogDensity ? seed_position.q : seed_position
        pg = DynamicHMC.evaluate_ℓ(lpdf, collect(float.(q0)); strict=true)
        n_written = 0
        eff_n_draws = n_draws
        fresh = true
        samples = _open_mmap(samples_path, Float64, (dimension, eff_n_draws); fresh)
        ring = _ring_create(ring_path, rng, dimension)
    end
    # Per-draw sidecars (each its own file), opened/grown in lockstep with the
    # samples file. On resume they keep their already-written columns.
    rng_states = _open_mmap(_rng_states_path(samples_path), UInt64, (_XOSHIRO_WORDS, eff_n_draws); fresh)
    divergences = _open_mmap(_divergences_path(samples_path), Int8, (eff_n_draws,); fresh)
    start_time = time_ns()
    try
        pg, rng, n_divergent, k = _stream_loop!(rng, lpdf, pg, kinetic_energy, stepsize;
            n_draws=eff_n_draws, max_tree_depth, ring, samples, rng_states, divergences,
            dimension, n_written, durable, progress, description, start_time)
        Mmap.sync!(samples); Mmap.sync!(rng_states); Mmap.sync!(divergences)
        ess = k > 10 ? _stream_ess(samples, k, dimension) : fill(NaN, dimension)
        _stream_result(samples_path, samples, rng_states, divergences,
            dimension, k, n_divergent, ess, rng, pg.q)
    finally
        close(ring.io)
    end
end

# Refuse to silently clobber an existing samples file that has no valid ring —
# the same "don't guess" discipline as `resolve_checkpoint_dir`.
_guard_fresh_over_existing(samples_path, ring_path, overwrite) = begin
    (overwrite || !isfile(samples_path) || !isnothing(_ring_read(ring_path))) && return
    throw(ArgumentError("""
    $(repr(samples_path)) already exists but has no valid safe ring to resume from
    ($(repr(ring_path)) is missing or corrupt). Refusing to guess. Pass
    overwrite=true to discard it and start fresh, or point `path` elsewhere."""))
end

"""
    stream_mcmc(rng, lpdf, position; path, n_draws, metric, stepsize, kwargs...)
    stream_mcmc(lpdf; path, n_draws, metric, stepsize, kwargs...)               # resume-only
    stream_mcmc(checkpoint, lpdf; path, n_draws, kwargs...)                     # seed from a WarmupHMC checkpoint
    stream_mcmc(checkpoint, lpdf, position; path, n_draws, rng, kwargs...)      # checkpoint kernel, explicit start
    stream_mcmc(checkpoint, lpdf, positions; path, n_draws, rngs, parallel)     # N chains under path/chain_<i>
    stream_mcmc(rngs, lpdf, positions; path, n_draws, metric, stepsize)         # N chains, explicit kernel

Fixed-kernel, interruptible, resumable NUTS sampling — pure sampling with NO
warm-up and NO adaptation. Streams draws into the mmappable `Float64` file at
`path` (zeroed at creation, draws as COLUMNS, `dimension × n_draws`, in the
sampler's raw SOURCE frame) while continuously persisting only `(rng,
last-safe-draw, dimension)` into a crash-safe "safe ring" at `path * ".ring"` —
the current position is NOT re-stored, it is read back from the samples file on
resume. A process killed mid-sample or mid-write resumes from the immediately
preceding state.

# Resuming
Pass the same `path`. If it holds a valid ring it is RESUMED: the rng comes from
the ring, the position is read from the file's last safe column, and only the
kernel hyperparameters are taken from the call. `n_draws` is a FLOOR — resuming
with a larger value extends the file (new columns zeroed) and keeps sampling; a
value at or below what is already written returns the existing draws.
`overwrite=true` discards an existing run and starts fresh. A crash inside the
very first transition (before draw 1) leaves an unresumable file that simply
restarts from the caller's seed — no draw is ever lost.

# Kernel hyperparameters (never persisted — supplied every call, or read from a checkpoint)
- `metric` — the mass-matrix SCALE `L` (`L*L'` ≈ posterior covariance): a
  `Diagonal`/matrix/WarmupHMC matrix expression, a vector of per-coordinate
  scales (std devs), or a ready `DynamicHMC.GaussianKineticEnergy`. A warm-up
  scale (`checkpoint`'s `scale_options[active_transformation]`) plugs in directly.
- `stepsize` — the fixed leapfrog step size.
- `max_tree_depth=10` — NUTS max tree depth.

# Streaming config
- `path` — the mmappable samples file. Required.
- `n_draws` — number of draws (sizes the zeroed file); a floor on resume.
- `durable=false` — `false` streams to the page cache (survives a PROCESS
  crash, the resume contract). `true` `Mmap.sync!`s each draw to disk before the
  ring points at it (survives a MACHINE/power crash too, at one msync per draw).
- `overwrite=false` — start fresh over an existing `path`.
- `progress=nothing` / `description="stream_mcmc"` — a Treebars node to hang a
  progress tree under (as `adaptive_warmup_mcmc`). It reports only the NON-fixed
  quantities — draws done, this session's draw rate, divergences, mean leapfrog
  steps per draw, and bulk-ESS recomputed on exponential (doubling) windows. The
  step size and metric are fixed and deliberately not shown. `nothing` runs with
  zero progress/ESS overhead.

# Output frame
The file and the returned `draws` are the SOURCE (working) frame — the sampler's
own parametrization, which is what resume reads back. Constrained model-frame
draws are produced on READ: `open_stream(path, lpdf; model=true)`, or
`WarmupHMC.back_transform`. For a plain log density the frames coincide, so the
file is already the posterior draws.

# Seeding from a WarmupHMC checkpoint
`stream_mcmc(payload, lpdf; ...)` (or a `cp_*.jls` path) reads the RNG, current
position, mass-matrix scale and step size out of the payload — the final adapted
kernel of a warm-up run — restores its reparametrization centerings onto `lpdf`,
and streams from there. `lpdf` must be built exactly as the run was given it.

Passing an explicit `position` keeps the checkpoint's KERNEL but overrides its
start point (e.g. an equidistant warm-up draw), with `rng` inheriting the
checkpoint's unless given. Passing a VECTOR of `positions` fans out one
independent chain per start point, each an interruptible/resumable stream under
`path/chain_<i>` (+ its `.ring`); re-calling the same `path` resumes every chain.
`parallel=true` threads them under a shared progress bar; `rngs` defaults to N
independent streams derived from the checkpoint's rng; `lpdf` is deepcopied per
chain (pass a length-N vector of lpdfs to override) so a stateful wrapper never
races. `stream_mcmc(rngs, lpdf, positions; metric, stepsize, ...)` is the same
fan-out with a hand-supplied kernel. Multi-chain returns a `Vector` of per-chain
result NamedTuples.

Returns a `NamedTuple`: `path`, `draws` (a view of the `dimension × n_drawn`
mmapped source-frame draws), `samples` (the live mapping), `n_drawn`,
`n_divergent` (of this call), `divergences` (a length-`n_drawn` `Int8` 0/1 view of
EVERY draw's divergence flag — their sum is the total), `rng_states` (a
`5 × n_drawn` `UInt64` view of every draw's post-draw Xoshiro state), `ess`
(per-coordinate bulk-ESS, sorted ascending; `NaN` for `<= 10` draws), `dimension`,
`rng`, `position`. Reopen a finished or partial run with [`open_stream`](@ref).

Reproducibility of resume-equals-uninterrupted is byte-for-byte for a fixed seed
only under single-threaded BLAS (`LinearAlgebra.BLAS.set_num_threads(1)`).
"""
stream_mcmc(rng, lpdf, position::AbstractVector;
        path, n_draws::Integer, metric, stepsize::Real, max_tree_depth::Integer=10,
        overwrite::Bool=false, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    dimension = LogDensityProblems.dimension(lpdf)
    length(position) == dimension || throw(DimensionMismatch(
        "initial `position` has length $(length(position)) but the lpdf has " *
        "dimension $dimension."))
    ring_path = _ring_path(path)
    resumable = !overwrite && _is_resumable(path, ring_path)
    resumable || _guard_fresh_over_existing(path, ring_path, overwrite)
    _stream_impl(lpdf;
        samples_path=path, ring_path, resumable,
        seed_rng=rng, seed_position=position,
        kinetic_energy=_stream_energy(metric), stepsize=float(stepsize),
        n_draws, max_tree_depth, durable, progress, description)
end

# Resume-only: the rng comes from the ring and the position from the file; the
# kernel is re-supplied by the caller.
stream_mcmc(lpdf; path, n_draws::Integer, metric, stepsize::Real,
        max_tree_depth::Integer=10, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    ring_path = _ring_path(path)
    _is_resumable(path, ring_path) || throw(ArgumentError(
        "no resumable run at $(repr(path)) (need both the samples file and a valid " *
        "$(repr(ring_path)) with at least one committed draw). Use the " *
        "`(rng, lpdf, position; ...)` form to start fresh."))
    _stream_impl(lpdf;
        samples_path=path, ring_path, resumable=true,
        seed_rng=nothing, seed_position=nothing,
        kinetic_energy=_stream_energy(metric), stepsize=float(stepsize),
        n_draws, max_tree_depth, durable, progress, description)
end

# Seed from a WarmupHMC checkpoint payload: read rng/position/scale/stepsize out
# of it. An already-resumable `path` continues that streaming run instead.
stream_mcmc(payload::NamedTuple, lpdf; path, n_draws::Integer,
        max_tree_depth::Integer=10, overwrite::Bool=false, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    kinetic_energy, stepsize = _checkpoint_kernel!(payload, lpdf)
    ring_path = _ring_path(path)
    resumable = !overwrite && _is_resumable(path, ring_path)
    resumable || _guard_fresh_over_existing(path, ring_path, overwrite)
    _stream_impl(lpdf;
        samples_path=path, ring_path, resumable,
        seed_rng=payload.rng, seed_position=payload.position_and_gradient,
        kinetic_energy, stepsize,
        n_draws, max_tree_depth, durable, progress, description)
end

# Checkpoint kernel, EXPLICIT start position. Bruno's case: take the hyperparameters
# (scale/stepsize) FROM the checkpoint, but start each chain from a supplied point
# (e.g. an equidistant warm-up draw) instead of the checkpoint's single saved
# position. `rng=nothing` inherits the checkpoint's rng; pass one per chain to fork
# the streams. An already-resumable `path` still continues that streaming run.
stream_mcmc(payload::NamedTuple, lpdf, position::AbstractVector; path, n_draws::Integer,
        rng=nothing, max_tree_depth::Integer=10, overwrite::Bool=false, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    kinetic_energy, stepsize = _checkpoint_kernel!(payload, lpdf)
    length(position) == payload.dimension || throw(DimensionMismatch(
        "start `position` has length $(length(position)) but the checkpoint holds a " *
        "$(payload.dimension)-dimensional problem."))
    ring_path = _ring_path(path)
    resumable = !overwrite && _is_resumable(path, ring_path)
    resumable || _guard_fresh_over_existing(path, ring_path, overwrite)
    _stream_impl(lpdf;
        samples_path=path, ring_path, resumable,
        seed_rng = isnothing(rng) ? payload.rng : rng, seed_position = position,
        kinetic_energy, stepsize,
        n_draws, max_tree_depth, durable, progress, description)
end

stream_mcmc(checkpoint_path::AbstractString, lpdf; kwargs...) =
    stream_mcmc(deserialize(checkpoint_path)::NamedTuple, lpdf; kwargs...)
stream_mcmc(checkpoint_path::AbstractString, lpdf, position::AbstractVector; kwargs...) =
    stream_mcmc(deserialize(checkpoint_path)::NamedTuple, lpdf, position; kwargs...)

# ---------------------------------------------------------------- multi-chain
# Run one chain per start position, each an independent interruptible/resumable
# stream under `path/chain_<i>` (+ its `.ring`), mirroring the adaptive sampler's
# `_chain_dir` layout. Re-calling with the same `path` resumes every chain from
# its own file. `parallel=true` threads the chains; each hangs its own progress
# sub-node under a shared bar, exactly as `adaptive_warmup_mcmc` does. Returns a
# `Vector` of the per-chain result NamedTuples.
_chain_stream_path(path, i) = joinpath(path, "chain_$i")

_stream_multichain(run_chain, n_chains; path, parallel, progress, description) = begin
    isnothing(path) && throw(ArgumentError("multi-chain stream_mcmc needs a base `path` directory"))
    mkpath(path)
    with_progress(progress, n_chains; description) do prog
        rv = Vector{Any}(missing, n_chains)
        if parallel
            Threads.@threads for i in 1:n_chains
                rv[i] = run_chain(i, prog, _chain_stream_path(path, i), string(description, ".", i))
                update_progress!(prog)
            end
        else
            for i in 1:n_chains
                rv[i] = run_chain(i, prog, _chain_stream_path(path, i), string(description, ".", i))
                update_progress!(prog)
            end
        end
        identity.(rv)
    end
end

# Multi-chain, checkpoint kernel + N start positions. `rngs=nothing` derives N
# independent streams from the checkpoint's rng; `lpdf` is deepcopied per chain
# (pass a length-N vector of lpdfs to override) so a stateful wrapper never races.
stream_mcmc(payload::NamedTuple, lpdf, positions::AbstractVector{<:AbstractVector};
        path, n_draws::Integer, rngs=nothing, parallel::Bool=true, max_tree_depth::Integer=10,
        overwrite::Bool=false, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    n_chains = length(positions)
    n_chains >= 1 || throw(ArgumentError("`positions` must be non-empty"))
    chain_rngs = isnothing(rngs) ? _derive_chain_rngs(payload.rng, n_chains) : rngs
    length(chain_rngs) == n_chains || throw(DimensionMismatch(
        "got $(length(chain_rngs)) rngs for $n_chains chains"))
    lpdfs = _stream_lpdfs(lpdf, n_chains)
    _stream_multichain(n_chains; path, parallel, progress, description) do i, prog, chain_path, chain_desc
        stream_mcmc(payload, lpdfs[i], positions[i]; path=chain_path, n_draws, rng=chain_rngs[i],
            max_tree_depth, overwrite, durable, progress=prog, description=chain_desc)
    end
end

stream_mcmc(checkpoint_path::AbstractString, lpdf, positions::AbstractVector{<:AbstractVector}; kwargs...) =
    stream_mcmc(deserialize(checkpoint_path)::NamedTuple, lpdf, positions; kwargs...)

# Multi-chain, EXPLICIT kernel + N start positions: one rng per chain, a hand-
# supplied `metric`/`stepsize`. Same per-chain layout, threading and deepcopy.
stream_mcmc(rngs::AbstractVector, lpdf, positions::AbstractVector{<:AbstractVector};
        path, n_draws::Integer, metric, stepsize::Real, parallel::Bool=true,
        max_tree_depth::Integer=10, overwrite::Bool=false, durable::Bool=false,
        progress=nothing, description::AbstractString="stream_mcmc") = begin
    n_chains = length(positions)
    length(rngs) == n_chains || throw(DimensionMismatch(
        "got $(length(rngs)) rngs for $n_chains chains"))
    lpdfs = _stream_lpdfs(lpdf, n_chains)
    _stream_multichain(n_chains; path, parallel, progress, description) do i, prog, chain_path, chain_desc
        stream_mcmc(rngs[i], lpdfs[i], positions[i]; path=chain_path, n_draws, metric, stepsize,
            max_tree_depth, overwrite, durable, progress=prog, description=chain_desc)
    end
end

"""
    open_stream(path) -> NamedTuple
    open_stream(path, lpdf; model=false) -> NamedTuple

Reopen an `stream_mcmc` run (finished or in-progress) for READING. Reads
`(rng, n_written, dimension)` from the safe ring at `path * ".ring"`, then
memory-maps the samples file and returns `draws` — a `dimension × n_drawn` view,
draws as COLUMNS — plus `samples` (the read-only mapping), `n_drawn`,
`dimension`, `rng`, and the source-frame `position` (the last drawn column). Also
returns the per-draw sidecars `divergences` (a length-`n_drawn` `Int8` 0/1 view)
and `rng_states` (a `5 × n_drawn` `UInt64` view of the post-draw Xoshiro state) —
each `nothing` for a run that predates them.

The stored draws are the SOURCE (working) frame. Pass `lpdf` and `model=true` to
get constrained model-frame draws instead (`draws` is then a fresh
`Matrix{Float64}`, back-transformed exactly as `WarmupHMC.back_transform`); with
`model=false` the `lpdf` is used only to check the dimension. Throws if `path`
has no readable safe ring.
"""
# Read-only mmap of an existing sidecar, sliced to its first `n` columns; the
# array's last dimension is `n_cols`. `nothing` if the file is absent (a run
# created before the sidecar existed).
_read_sidecar(path, ::Type{T}, lead_dims, n_cols, n) where {T} = begin
    isfile(path) || return nothing
    dims = (lead_dims..., n_cols)
    arr = Mmap.mmap(open(path, "r"), Array{T,length(dims)}, dims)
    @view arr[ntuple(_ -> Colon(), length(lead_dims))..., 1:n]
end

open_stream(path, lpdf=nothing; model::Bool=false) = begin
    ring_path = _ring_path(path)
    r = _ring_read(ring_path)
    isnothing(r) && throw(ArgumentError(
        "no readable safe ring at $(repr(ring_path)); $(repr(path)) is not a " *
        "resumable stream_mcmc run."))
    _, rng, n_written, dimension = r
    if !isnothing(lpdf)
        LogDensityProblems.dimension(lpdf) == dimension || throw(DimensionMismatch(
            "the run at $(repr(path)) is $dimension-dimensional but the supplied " *
            "lpdf has dimension $(LogDensityProblems.dimension(lpdf))."))
    end
    model && isnothing(lpdf) && throw(ArgumentError(
        "model=true needs the `lpdf` that produced the run to back-transform."))
    total_cols = filesize(path) ÷ (sizeof(Float64) * dimension)
    io = open(path, "r")
    samples = Mmap.mmap(io, Matrix{Float64}, (dimension, total_cols))
    close(io)
    source = @view samples[:, 1:n_written]
    draws = source
    if model
        # `reparametrize!` maps source -> model IN PLACE and returns nothing, so
        # copy first (also: never mutate the read-only mapping), then hand back
        # the mutated dense matrix.
        m = Matrix{Float64}(source)
        reparametrize!(lpdf, m)
        draws = m
    end
    # Sidecars are sized to whatever the on-disk file holds; slice to n_written.
    rng_total = isfile(_rng_states_path(path)) ?
        filesize(_rng_states_path(path)) ÷ (sizeof(UInt64) * _XOSHIRO_WORDS) : 0
    div_total = isfile(_divergences_path(path)) ? filesize(_divergences_path(path)) : 0
    rng_states = _read_sidecar(_rng_states_path(path), UInt64, (_XOSHIRO_WORDS,), rng_total, n_written)
    divergences = _read_sidecar(_divergences_path(path), Int8, (), div_total, n_written)
    (; path, draws, samples, n_drawn=n_written, divergences, rng_states,
       dimension, rng, position=collect(@view samples[:, n_written]))
end
