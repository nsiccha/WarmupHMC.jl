using TestItemRunner

# `Enzyme` is NOT an idle import in the test items. `AutoEnzyme()` is a BACKEND
# HANDLE, not a backend: DifferentiationInterface can only differentiate through
# it when Enzyme itself is loaded in the session. `invariant_scoring.jl` and
# `wrapped_logdensity.jl` construct one directly. Remove it and those items do
# not go quiet — they error.
#
# Enzyme is the only backend this suite exercises, deliberately. The initializer
# does not need one: `mypathfinder` pins `adtype = NoAD()` precisely so
# Optimization never synthesizes a gradient of its own, so no second AD package
# is reachable from the tested paths.

@run_package_tests()
