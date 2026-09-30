@testitem "completion init runs Pathfinder per chain" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra, Statistics
    using LogDensityProblems
    using Distributions, Serialization

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -t4 -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "completion_init.jl"))'
#
# Pins the verified contract behind `warmuphmc-use` §2/§5f: `completion_warmup_mcmc`
# initializes EVERY chain with Pathfinder, exactly as `adaptive_warmup_mcmc` does —
# by default, for a per-chain `init` vector mixing positions and `missing`, and on
# resume for a slot that never checkpointed.

struct CompletionInitCorrGaussian{V<:AbstractVector,M<:AbstractMatrix}
    mu::V
    Sigma::M
    invSigma::M
    logdet::Float64
end
function CompletionInitCorrGaussian(mu, Sigma)
    F = cholesky(Symmetric(Sigma))
    CompletionInitCorrGaussian(mu, Sigma, inv(F), logdet(F))
end
LogDensityProblems.capabilities(::Type{<:CompletionInitCorrGaussian}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::CompletionInitCorrGaussian) = length(p.mu)
function LogDensityProblems.logdensity_and_gradient(p::CompletionInitCorrGaussian, x)
    r = x .- p.mu
    (-dot(r, p.invSigma * r) / 2 - p.logdet / 2 - length(x) / 2 * log(2π),
        -(p.invSigma * r))
end
LogDensityProblems.logdensity(p::CompletionInitCorrGaussian, x) =
    first(LogDensityProblems.logdensity_and_gradient(p, x))

# A correlated target: Pathfinder's fitted covariance then has genuine
# off-diagonal mass, while a skipped Pathfinder (diagonal fallback) would leave
# the init scale exactly diagonal. Off-diagonal energy in the pristine :init
# scale therefore proves the optimizer ran.
_init_lpdf = CompletionInitCorrGaussian(
    [4.0, -3.0, 8.0],
    Matrix([1.0 0.0 0.0; 0.9 0.45 0.0; 0.2 -0.3 1.1] *
        [1.0 0.0 0.0; 0.9 0.45 0.0; 0.2 -0.3 1.1]'),
)
_init_offdiag(F) = norm(F - Diagonal(diag(F)))

# Pristine post-init state per chain: the :init callback fires on the state as
# `initialize_mcmc` left it, before any warm-up window can adapt the scales.
function _init_recorder()
    records = Any[]
    lock = ReentrantLock()
    callback = (state, stage) -> begin
        if stage === :init
            Base.lock(lock) do
                push!(records, (;
                    position=copy(state.position),
                    pf_m1=Matrix(state.scale_options.pathfinder.m1),
                    active=state.active_transformation,
                ))
            end
        end
        false
    end
    records, callback
end

const _INIT_SEED = 5000
const _INIT_N = 2
const _INIT_DRAWS = 40
_init_rngs() = [Xoshiro(_INIT_SEED + i) for i in 1:_INIT_N]

@testset "default init matches adaptive per chain" begin
    records, callback = _init_recorder()
    dir = mktempdir()
    TB = WarmupHMC.Treebars
    out = TB.with_progress(:state; description="init-default") do progress
        completion_warmup_mcmc(_init_rngs(), _init_lpdf; min_completed=_INIT_N,
            n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
            progress, monitor_ess=true, callback, checkpoint_dir=dir)
    end
    @test out.completion.completed_chain_indices == [1, 2]
    # Every chain initialized, through Pathfinder, from its own RNG slot.
    @test length(records) == _INIT_N
    for record in records
        @test record.active === :pathfinder
        @test _init_offdiag(record.pf_m1) > 0.1
    end
    # Each record belongs to exactly one returned chain (frozen `position`).
    for record in records
        @test count(r -> r.initial_position == record.position, out.results) == 1
    end
    # CP-0 agrees with the callback record per chain.
    for i in 1:_INIT_N
        payload = deserialize(joinpath(dir, "chain_$i", "cp_init.jls"))
        @test any(r -> r.position == payload.position, records)
    end
    # Bit-identical to the same-seed single-chain adaptive init.
    for i in 1:_INIT_N
        ctrl_records, ctrl_callback = _init_recorder()
        adaptive_warmup_mcmc(Xoshiro(_INIT_SEED + i), _init_lpdf;
            n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
            monitor_ess=false, callback=ctrl_callback)
        hit = filter(r -> r.position == ctrl_records[1].position, records)
        @test length(hit) == 1
        @test hit[1].pf_m1 == ctrl_records[1].pf_m1
    end
end

@testset "per-chain init vector routes per chain" begin
    x0 = [20.0, 20.0, 20.0]
    records, callback = _init_recorder()
    out = completion_warmup_mcmc(_init_rngs(), [_init_lpdf, _init_lpdf];
        min_completed=_INIT_N, n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback, init=[x0, missing])
    @test length(records) == _INIT_N
    by_chain = Dict{Int,Any}()
    for record in records
        idxs = findall(r -> r.initial_position == record.position, out.results)
        @test length(idxs) == 1
        by_chain[out.results[idxs[1]].chain_index] = record
    end
    x_records, x_callback = _init_recorder()
    adaptive_warmup_mcmc(Xoshiro(_INIT_SEED + 1), _init_lpdf;
        n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback=x_callback, init=x0)
    m_records, m_callback = _init_recorder()
    adaptive_warmup_mcmc(Xoshiro(_INIT_SEED + 2), _init_lpdf;
        n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback=m_callback)
    @test by_chain[1].position == x_records[1].position
    @test by_chain[1].pf_m1 == x_records[1].pf_m1
    @test by_chain[2].position == m_records[1].position
    @test by_chain[2].pf_m1 == m_records[1].pf_m1
    @test by_chain[1].position != by_chain[2].position
end

@testset "resume initializes a never-checkpointed slot" begin
    _, fresh_callback = _init_recorder()
    dir = mktempdir()
    fresh = completion_warmup_mcmc(_init_rngs(), _init_lpdf;
        min_completed=_INIT_N, n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback=fresh_callback, checkpoint_dir=dir)
    @test fresh.completion.completed_chain_indices == [1, 2]
    # Model an interrupted run: chain 1 finished and checkpointed, chain 2 never
    # started, the process died before the terminal write.
    rm(joinpath(dir, "completion_terminal.jls"))
    rm(joinpath(dir, "chain_2"); recursive=true)
    records, callback = _init_recorder()
    out = completion_warmup_mcmc(_init_rngs(), _init_lpdf;
        min_completed=_INIT_N, n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback, checkpoint_dir=dir, resume=true)
    @test out.completion.completed_chain_indices == [1, 2]
    # Chain 1 restores from its checkpoint (no re-init); chain 2 initializes.
    @test length(records) == 1
    idxs = findall(r -> r.initial_position == records[1].position, out.results)
    @test length(idxs) == 1
    @test out.results[idxs[1]].chain_index == 2
    @test records[1].active === :pathfinder
    @test _init_offdiag(records[1].pf_m1) > 0.1
    ctrl_records, ctrl_callback = _init_recorder()
    adaptive_warmup_mcmc(Xoshiro(_INIT_SEED + 2), _init_lpdf;
        n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, callback=ctrl_callback)
    @test records[1].position == ctrl_records[1].position
    @test records[1].pf_m1 == ctrl_records[1].pf_m1
    @test isfile(joinpath(dir, "chain_2", "cp_init.jls"))
end

@testset "completion draws equal adaptive draws per chain" begin
    out_c = completion_warmup_mcmc(_init_rngs(), _init_lpdf;
        min_completed=_INIT_N, n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false)
    out_a = adaptive_warmup_mcmc(_init_rngs(), _init_lpdf;
        n_draws=_INIT_DRAWS, target_acceptance_rate=0.8,
        monitor_ess=false, parallel=true)
    for i in 1:_INIT_N
        @test out_c.results[i].chain_index == i
        @test out_c.results[i].initial_position == out_a[i].initial_position
        @test out_c.results[i].posterior_position == out_a[i].posterior_position
        @test out_c.results[i].n_divergent_samples == out_a[i].n_divergent_samples
    end
end
end
