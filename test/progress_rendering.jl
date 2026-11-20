@testitem "progress labels stay compact" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, LinearAlgebra, Random
    using LogDensityProblems
    const DynamicHMC = WarmupHMC.DynamicHMC
    const Pathfinder = WarmupHMC.Pathfinder

# Focused invocation from the package root (test environment instantiates per
# .github/workflows/test.yml; `@run_package_tests` from `-e` mis-discovers, so
# pass the root explicitly):
# julia --startup-file=no --project=test -e 'using TestItemRunner;
# TestItemRunner.run_tests(pwd(); filter=ti->endswith(ti.filename, "progress_rendering.jl"))'

# The "active transformation" progress label renders the active scale through
# `short_string`. A factor type with no method fell through to `string` of the
# whole d×d matrix — after `f210206` switched the Pathfinder scale to a dense
# Cholesky factor, that was ~2.7M characters at d = 400, on every render.
label(L) = string(WarmupHMC.ActiveTransformation(
    DynamicHMC.GaussianKineticEnergy(WarmupHMC.MatrixFactorization(L, L'),
                                     WarmupHMC.MatrixInverse(L')),
    [1.0, 2.0]))

@testset "every init scale renders in O(1) characters" begin
    d = 200
    target = DiagGaussian(zeros(d), exp.(range(-1, 1, length = d)))
    fit = Pathfinder.pathfinder(target; rng = Xoshiro(1), ndraws = 10, progress = false)
    Σ = fit.fit_distribution.Σ
    @test Σ isa Pathfinder.WoodburyPDMat
    dense = Matrix(Σ)
    for (name, sq) in (("Pathfinder Woodbury", Σ), ("dense matrix", dense),
                       ("diagonal", Diagonal(diag(dense))))
        s = label(WarmupHMC._initial_pathfinder_scale(sq, d))
        @test length(s) < 200
        @test !occursin("0.0 0.0", s)
    end
    # A Pathfinder init keeps the low-rank factor and its `Pathfinder(k)` label;
    # only a caller-supplied dense matrix gets the dense Cholesky label.
    @test startswith(label(WarmupHMC._initial_pathfinder_scale(Σ, d)), "Pathfinder(")
    @test !occursin("dense", label(WarmupHMC._initial_pathfinder_scale(Σ, d)))
    @test startswith(label(WarmupHMC._initial_pathfinder_scale(dense, d)), "Pathfinder(dense, d = $d)")
    # The adaptive and diagonal options keep their existing labels.
    adaptive = WarmupHMC.MatrixFactorization(WarmupHMC.SuccessiveReflections(d), Diagonal(ones(d)))
    @test startswith(label(adaptive), "Adaptive(")
end
end

@testitem "warm-up restarts are visible on the chain progress bar" setup=[WarmupHMCSharedFixtures] begin
    using WarmupHMC, Random, Test

    target = DiagGaussian(zeros(3), [0.5, 1.0, 2.0])
    start = (; position=zeros(3), squared_scale=ones(3))
    settings = (; n_draws=100, stepsize_adaptation_limit=10, n_evaluations=40,
                 variance_cond_target=1.0, max_divergent_fraction=1.0)
    for mode in (:single, :multi, :completion)
        snapshots = Channel{Tuple{Int,String}}(128)
        function inspect_window(state, stage)
            if stage === :window && state.current_transition_counter == 0
                put!(snapshots, (state.outer_counter, _Treebars.render_text(state.progress)))
            end
            return false
        end
        root = _Treebars.with_progress(:state; description="probe") do progress
            kwargs = (; settings..., progress, callback=inspect_window)
            if mode === :single
                adaptive_warmup_mcmc(Xoshiro(1), target; init=start, kwargs...)
            elseif mode === :multi
                adaptive_warmup_mcmc(Xoshiro.(1:2), target; init=fill(start, 2), kwargs...)
            else
                completion_warmup_mcmc(Xoshiro.(1:2), target; init=fill(start, 2),
                    stuck_stepsize_ratio=Inf, max_admitted_divergent_fraction=1.0,
                    max_admitted_log_mass_gap=Inf, kwargs...)
            end
            progress
        end
        close(snapshots)
        chains = mode === :single ? ["MCMC"] : ["MCMC.1", "MCMC.2"]
        seen = Set{String}()
        for (window, tree) in snapshots
            chain_line = first(split(tree, '\n'))
            names = filter(name -> occursin("$name (0/110)", chain_line), chains)
            @test length(names) == 1
            union!(seen, names)
            @test occursin("warm-up restarted after window $window", chain_line)
        end
        @test seen == Set(chains)
        final_tree = _Treebars.render_text(root)
        for name in chains
            @test count("$name (110/110)", final_tree) == 1
        end
        @test occursin("divergent:", final_tree)
        @test !occursin("warm-up restarted", final_tree)
        @test !occursin("✗", final_tree)
    end
end
