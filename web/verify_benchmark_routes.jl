# Verify that every benchmark results JSON renders through the app.
#
#     julia --project=web web/verify_benchmark_routes.jl
#
# Exits nonzero on the first failure, so it is usable as a CI step.
#
# WHY THIS IS A SEPARATE SCRIPT AND NOT A `@testitem`
# ---------------------------------------------------
# The main suite runs under `web/src/test/Project.toml`, which deliberately does
# not depend on `WarmupHMCWeb` (and so has no HTMXObjects, no JSON, no
# BridgeStan, no PosteriorDB). Adding them would drag the whole web stack into
# the nine-row matrix to test a renderer. So this lives beside the app and runs
# against the app's own environment.
#
# WHAT IT PROTECTS
# ----------------
# `docs/benchmark/results/*.json` is the single source for the evidence tables.
# The app and docs both project the shared semantic value; the app also records
# its interactive routes for the static gallery. A driver that writes a shape
# the renderer cannot parse would therefore break both consumers. This asserts
# the app half of the chain: every checked-in file loads, every file is reachable
# at a route, every route is listed for recording, and the index links to each
# one. The production docs build separately exercises the Markdown projection.

using WarmupHMCWeb
using HTMXObjects: HTTP

const PORT = parse(Int, get(ENV, "WHMC_VERIFY_PORT", "8099"))

failures = String[]
check(name, cond) = cond ? println("  ok    $name") :
                           (println("  FAIL  $name"); push!(failures, name))

keys = WarmupHMCWeb.benchmark_keys()
println("Found $(length(keys)) results file(s) under docs/benchmark/results/")
check("at least one results file is checked in", !isempty(keys))

# --- Every file parses, and parses into something renderable ---
println("\nLoading each results file:")
for key in keys
    loaded = try
        WarmupHMCWeb.benchmark_load(key)
    catch err
        check("$key loads", false)
        println("        $err")
        continue
    end
    check("$key loads", true)
    # A file with no tables renders as a bare provenance list — almost certainly
    # a driver bug (an empty run, or rows under an unexpected key), and exactly
    # the silent-empty-page case this script exists to catch.
    check("$key has at least one table", !isempty(loaded.tables))
    for (name, table) in loaded.tables
        check("$key/$name has rows and columns",
              !isempty(table) && !isempty(first(table)))
    end
end

# --- Every file is scheduled for static recording into the docs ---
println("\nRecording coverage:")
paths = WarmupHMCWeb.APPDATA.recording_paths
check("/benchmarks is recorded", "/benchmarks" in paths)
for key in keys
    check("/benchmark/$key is recorded", "/benchmark/$key" in paths)
end

# --- Every route actually serves, and the index links to each file ---
println("\nServing routes on port $PORT:")
WarmupHMCWeb.serve(; host="127.0.0.1", port=PORT, async=true)
sleep(6)

fetch(path) = try
    r = HTTP.get("http://127.0.0.1:$PORT$path"; status_exception=false, readtimeout=120)
    (r.status, String(r.body))
catch err
    (0, string(err))
end

status, index = fetch("/benchmarks")
check("/benchmarks returns 200", status == 200)
for key in keys
    st, body = fetch("/benchmark/$key")
    check("/benchmark/$key returns 200", st == 200)
    check("/benchmark/$key renders a table", occursin("<table", body))
    check("/benchmarks links to $key", occursin("/benchmark/$key\"", index))
end

# A results file that is not on disk must degrade to an empty state, not a 500 —
# the index is rendered from the same directory listing, but a stale recorded
# page or a hand-typed URL can outlive the file it names.
st, body = fetch("/benchmark/definitely~not~a~file")
check("missing file degrades gracefully",
      st == 200 && occursin("No results file", body))

if isempty(failures)
    println("\nAll checks passed ($(length(keys)) results file(s)).")
else
    println("\n$(length(failures)) FAILURE(S):")
    foreach(f -> println("  - $f"), failures)
    exit(1)
end
