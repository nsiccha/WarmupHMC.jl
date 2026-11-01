# Run in the package test/consumer environment. The input construction is outside
# the measurement; every fit gets the same reflection and iteration budgets.
using WarmupHMC, Random, LinearAlgebra, Statistics

BLAS.set_num_threads(1)
println("dimension,draws,reflections,maxiter,median_seconds,allocated_bytes,retained_bytes,iterations")
for (d, n) in ((512, 64), (1024, 64), (2048, 64), (512, 128), (512, 256))
    draws = randn(Xoshiro(700 + d + n), d, n)
    weights = ones(n)
    fit() = fit_gaussian_moments(draws; weights, max_reflections=3, maxiter=4)
    result = fit()
    bytes = @allocated fit()
    times = [@elapsed fit() for _ in 1:9]
    println(join((d, n, result.diagnostics.reflections, 4, median(times), bytes,
                  Base.summarysize(result), sum(result.diagnostics.iterations)), ','))
end
