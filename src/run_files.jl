# Files a sampler writes beside its checkpoints, shared by every sampler that
# writes them: the checkpoint schema version and the write-once JSON run records.

# --- Run manifest / summary ----------------------------------------------------
#
# Write-once JSON at the checkpoint_dir root, never rewritten:
#
#   run_manifest.json  at run START    — run identity + the criteria in force
#   run_summary.json   at FINALIZE     — the terminal answer
#
# The list above is the count; it is not also stated as a number, because a
# number and a list that can disagree eventually do.
#
# EXISTENCE of run_summary.json IS the run-completed signal: a consumer polling
# for "is this run done?" needs no parsing and no field. An earlier draft had a
# single manifest rewritten at finalize; that reintroduces the cp_latest.jls
# torn-read hazard in the one file every consumer must read to identify a run,
# so it was rejected in review.
#
# JSON rather than .jls lets a web process read run identity without Julia.
# JSON.jl owns encoding; non-finite numbers keep the on-disk null convention.

# Atomic write, same reason as the checkpoints: a crash during a WRITE-ONCE file
# must not strand a half-written one that a consumer would read as authoritative.
# `pairs` is deliberately Pair{String,Any} at every call site: a literal array
# mixing an Int with an Inf promotes to Float64, which would silently render
# `schema_version` as `1.0` and break a consumer parsing it as an integer.
_write_json(path, pairs) = begin
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path); cleanup=false)
    try
        JSON.json(io, Dict(pairs); allownan=true, nan="null", inf="null", ninf="null")
        close(io)
        mv(tmp, path; force=true)
    catch
        close(io); rm(tmp; force=true); rethrow()
    end
    nothing
end

"""
    checkpoint_schema_version() -> Int

Version of the shared boundary-checkpoint/manifest contract. The internal
`checkpoint_schema_version(Val(sampler))` also reports a sampler's maximum
readable version: adaptive v3 describes an unfinished window, while its
completed-boundary writer retains v2 compatibility.

Policy (the contract, not just the number):

* **Additive changes do NOT bump it.** New keys may appear at any time, so
  readers MUST ignore unknown keys.
* **Changing an existing key's meaning or type, or removing it, DOES bump it.**
  A bump means exactly "re-read the contract".
* Therefore a reader seeing a version NEWER than it knows should **fail loudly**:
  since additions never bump, a higher version can only mean existing semantics
  changed.
"""
checkpoint_schema_version() = 2

# Maximum readable version per sampler. Adaptive v3 can describe an unfinished
# window. Its completed-boundary writer still emits v2, preserving compatibility
# for old consumers and leaving other samplers' checkpoint contracts unchanged.
checkpoint_schema_version(::Val) = checkpoint_schema_version()
checkpoint_schema_version(::Val{:adaptive}) = 3
