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
# JSON rather than .jls so a web process can read run identity WITHOUT a Julia
# runtime — that is the whole point, and it is why this hand-rolls a tiny writer
# instead of taking a JSON dependency: the contents are a flat, fully-controlled
# set of scalars and small arrays, and WarmupHMC is heading for General
# registration where every added dep is a liability.

_json_esc(s) = replace(
    replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n"),
    r"[\x00-\x1f]" => c -> "\\u" * string(Int(only(c)); base=16, pad=4))
_json_val(::Nothing) = "null"
_json_val(x::Bool) = x ? "true" : "false"
_json_val(x::Real) = isfinite(x) ? string(x) : "null"   # Inf/NaN are not JSON
_json_val(x::Union{AbstractString,Symbol}) = "\"" * _json_esc(x) * "\""
_json_val(x::AbstractVector) = "[" * join(map(_json_val, x), ",") * "]"
_json_val(x) = _json_val(string(x))
_json_object(pairs) = "{" * join(
    [ "\"" * _json_esc(k) * "\":" * _json_val(v) for (k, v) in pairs ], ",") * "}"

# Atomic write, same reason as the checkpoints: a crash during a WRITE-ONCE file
# must not strand a half-written one that a consumer would read as authoritative.
# `pairs` is deliberately Pair{String,Any} at every call site: a literal array
# mixing an Int with an Inf promotes to Float64, which would silently render
# `schema_version` as `1.0` and break a consumer parsing it as an integer.
_write_json(path, pairs) = begin
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path); cleanup=false)
    try
        write(io, _json_object(pairs))
        close(io)
        mv(tmp, path; force=true)
    catch
        close(io); rm(tmp; force=true); rethrow()
    end
    nothing
end

"""
    checkpoint_schema_version() -> Int

Version of the on-disk checkpoint/manifest contract, shared by ALL samplers.

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
