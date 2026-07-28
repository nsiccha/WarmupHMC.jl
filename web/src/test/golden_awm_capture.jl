# Write the frozen golden baseline that `golden_awm.jl` compares against.
#
#     julia --project=web/src/test web/src/test/golden_awm_capture.jl
#
# `golden_awm.jls` is gitignored and never committed — it is platform- and
# BLAS-sensitive, so it is a LOCAL artifact you capture before a refactor and
# check after. Capturing it after a behaviour change simply pins the new
# behaviour; see the header of `golden_awm.jl` for what that does and does not
# buy you.
#
# This replaces the old `julia --project test/golden_awm.jl capture` mode. That
# mode branched on `abspath(PROGRAM_FILE) == abspath(@__FILE__)` and read its
# argument from `get(ARGS, 1, "check")` — under TestItemRunner `ARGS` carries the
# test SELECTORS, so `--skip-tag=enzyme` would have been parsed as a mode. The
# old `check` mode is gone: it duplicated what the test item asserts, and the
# item reports the same diffs.

using ForwardDiff, DifferentiationInterface
using LinearAlgebra: BLAS

# Same pin as the suite's `Determinism` snippet — a baseline captured under
# multithreaded BLAS could not be reproduced byte-for-byte by a run under it.
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "targets.jl"))
include(joinpath(@__DIR__, "golden_awm_common.jl"))

out = run_all()
serialize(GOLDEN_PATH, out)
println("CAPTURED baseline -> $GOLDEN_PATH")
for (name, o) in pairs(out)
    println("  [$name] draws=", size(o.result.posterior_position),
            " active=", o.result.active_transformation,
            " restarts=", length(o.result.scale_changes))
end
