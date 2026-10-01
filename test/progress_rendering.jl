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
    # The dense Cholesky factor gets its own name.
    @test startswith(label(WarmupHMC._initial_pathfinder_scale(Σ, d)), "Pathfinder(dense, d = $d)")
    # The adaptive and diagonal options keep their existing labels.
    adaptive = WarmupHMC.MatrixFactorization(WarmupHMC.SuccessiveReflections(d), Diagonal(ones(d)))
    @test startswith(label(adaptive), "Adaptive(")
end
end
