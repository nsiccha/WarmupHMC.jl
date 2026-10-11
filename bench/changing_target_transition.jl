# In-memory versus file-backed fixed-kernel NUTS for a changing 2D target.
# Run after resolving the test environment (see .github/workflows/test.yml):
#   julia --project=test bench/changing_target_transition.jl
# Six independent subjects each take 45 post-initialization transitions, the
# small SAEM shape that motivated this API. Both arms use identical targets,
# kernels, starts and RNGs; the final positions must match exactly. Timings
# exclude JIT compilation and the first transition, and include the ordinary
# per-call file reopening and summary work of stream_mcmc. They are a local
# overhead comparison, not posterior quality or portable sampler throughput.

using WarmupHMC, LogDensityProblems, Random, LinearAlgebra, Statistics

struct ChangingGaussian
    mu::Vector{Float64}
    sigma::Vector{Float64}
end
LogDensityProblems.capabilities(::Type{ChangingGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::ChangingGaussian) = length(p.mu)
LogDensityProblems.logdensity(p::ChangingGaussian, q) =
    -sum(abs2, (q .- p.mu) ./ p.sigma) / 2
function LogDensityProblems.logdensity_and_gradient(p::ChangingGaussian, q)
    z = (q .- p.mu) ./ p.sigma
    (-sum(abs2, z) / 2, -z ./ p.sigma)
end

const targets = [ChangingGaussian([0.02 * t, -0.01 * t], [0.7, 1.2]) for t in 1:46]
const kernel = (; metric=[0.7, 1.2], stepsize=0.5, max_tree_depth=8)
const n_subjects = 6

function memory_run()
    states = [FixedKernelNUTSState(Xoshiro(i), zeros(2); kernel...) for i in 1:n_subjects]
    for i in 1:n_subjects
        states[i] = fixed_kernel_nuts_step(states[i], targets[1]).state
    end
    elapsed = @elapsed for t in 2:length(targets), i in 1:n_subjects
        states[i] = fixed_kernel_nuts_step(states[i], targets[t]).state
    end
    elapsed, [copy(s.position) for s in states]
end

function stream_run()
    mktempdir() do dir
        paths = [joinpath(dir, "subject_$i") for i in 1:n_subjects]
        positions = Vector{Vector{Float64}}(undef, n_subjects)
        for i in 1:n_subjects
            stream_mcmc(Xoshiro(i), targets[1], zeros(2);
                path=paths[i], n_draws=1, kernel...)
        end
        elapsed = @elapsed for t in 2:length(targets), i in 1:n_subjects
            result = stream_mcmc(targets[t]; path=paths[i], n_draws=t, kernel...)
            positions[i] = copy(result.position)
        end
        elapsed, positions
    end
end

BLAS.set_num_threads(1)  # matched deterministic comparison
memory_run()
stream_run()
mem_times = Float64[]
file_times = Float64[]
for _ in 1:3
    GC.gc()
    mem_elapsed, mem_positions = memory_run()
    GC.gc()
    file_elapsed, file_positions = stream_run()
    @assert mem_positions == file_positions
    push!(mem_times, mem_elapsed)
    push!(file_times, file_elapsed)
end
println("steps=$(n_subjects * (length(targets) - 1)) memory_s=$(mem_times) stream_s=$(file_times)")
println("median_per_step_us memory=$(median(mem_times) * 1e6 / 270) stream=$(median(file_times) * 1e6 / 270) ratio=$(median(file_times) / median(mem_times))")
