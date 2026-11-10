@testitem "Callable accessors" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC, Random, LogDensityProblems
    using DifferentiationInterface, Enzyme

# Snag built-an-indexed-3fed939c. A `Reparametrization` accessor (location or
# log-scale argument) is a constant or a callable applied to the whole
# source-coordinate vector. Before 0.2.15 only a `Function` was called: a callable
# struct that did not subtype `Function` passed through `maybecall` as if it were a
# constant, and `reparam` then failed with a `MethodError` (fixed by commit
# `67bafc1`, snag regularised-hors-bcdcdd84). The scalar transform takes `Real`
# arguments, so construction now also refuses an accessor that is neither a `Real`
# nor callable. These tests pin the reporter's case end to end.

const PC = WarmupHMC.PartiallyCentered
const Rep = WarmupHMC.Reparametrization
const IR = WarmupHMC.IndexedReparametrization
const RP = WarmupHMC.ReparametrizedProblem

# The reporter's accessor, verbatim: one concrete type for "read coordinate i" and
# "use a constant", deliberately NOT a `Function` subtype.
struct Coordinate
    i::Int
    value::Float64
end
(c::Coordinate)(x) = c.i == 0 ? c.value : x[c.i]
# The same accessor as a `Function` subtype: the only spelling that worked before.
struct CoordinateF <: Function
    i::Int
    value::Float64
end
(c::CoordinateF)(x) = c.i == 0 ? c.value : x[c.i]
# A struct with no call method: neither a constant nor an accessor.
struct NotCallable
    i::Int
end

# A hierarchical target in its centered (model) frame, the reporter's shape: two
# blocks whose location and log-scale are sampled (coordinates 1:3 with 10, 11 and
# 4:6 with 12, 13) and one block with a fixed, known location 0.5 and scale 2
# (coordinates 7:9). Each of the nine effects has one Gaussian observation.
struct ThreeBlocks end
const BLOCKS = ((1:3, 10, 11), (4:6, 12, 13))
const FIXED = 7:9
const DATA = (0.8, -0.3, 1.1, 2.0, 1.7, 2.4, 0.1, 1.0, 0.6)
LogDensityProblems.dimension(::ThreeBlocks) = 13
LogDensityProblems.capabilities(::Type{ThreeBlocks}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(::ThreeBlocks, y)
    lp = zero(eltype(y))
    g = zero(y)
    for (block, mi, ti) in BLOCKS
        mu, tau = y[mi], y[ti]
        w = exp(-2tau)
        lp += -mu^2 / 2 - tau^2 / 2
        g[mi] += -mu
        g[ti] += -tau
        for i in block
            r = y[i] - mu
            lp += -r^2 * w / 2 - tau
            g[i] += -r * w
            g[mi] += r * w
            g[ti] += r^2 * w - 1
        end
    end
    for i in FIXED
        lp += -(y[i] - 0.5)^2 / 8
        g[i] += -(y[i] - 0.5) / 4
    end
    for i in eachindex(DATA)
        lp += -(y[i] - DATA[i])^2 / 2
        g[i] += -(y[i] - DATA[i])
    end
    lp, g
end
LogDensityProblems.logdensity(p::ThreeBlocks, y) = first(LogDensityProblems.logdensity_and_gradient(p, y))

# Sampled non-centered (source c = 0) against a centered model (target c = 1).
spec(A) = IR([
    [i => Rep(PC(1.0), PC(0.0), A(10, 0.0), A(11, 0.0)) for i in 1:3];
    [i => Rep(PC(1.0), PC(0.0), A(12, 0.0), A(13, 0.0)) for i in 4:6];
    [i => Rep(PC(1.0), PC(0.0), A(0, 0.5), A(0, log(2.0))) for i in FIXED]
])

# Tuples, not arrays: a `const` array re-evaluated with the item warns of a redefinition.
const X0 = (0.3, -0.7, 1.2, 0.4, -0.2, 0.9, -1.1, 0.5, 0.2, 0.6, -0.4, 1.3, 0.1)
X = collect(X0)

@testset "the target's analytic gradient" begin
    _, g = LogDensityProblems.logdensity_and_gradient(ThreeBlocks(), X)
    @test g ≈ fd_gradient(y -> LogDensityProblems.logdensity(ThreeBlocks(), y), X) rtol = 1e-6
end

@testset "maybecall: a number is a constant, every other value is called" begin
    v = [1.0, 7.0, 3.0]
    @test WarmupHMC.maybecall(0.5, v) === 0.5
    @test WarmupHMC.maybecall(2, v) === 2
    @test WarmupHMC.maybecall(x -> x[2], v) === 7.0
    @test WarmupHMC.maybecall(CoordinateF(3, 0.0), v) === 3.0
    @test WarmupHMC.maybecall(Coordinate(3, 0.0), v) === 3.0   # was: Coordinate(3, 0.0)
    @test WarmupHMC.maybecall(Coordinate(0, 0.25), v) === 0.25
end

@testset "a callable struct accessor maps like the closure it replaces" begin
    x = [0.9, 0.5, 0.2]
    closure = WarmupHMC.reparam(Rep(PC(1.0), PC(0.0), x -> x[2], x -> x[3]), x[1], x)
    # Before 0.2.15: MethodError: no method matching reparam(::PartiallyCentered{Float64},
    # ::PartiallyCentered{Float64}, ::Float64, ::Coordinate, ::Coordinate).
    @test WarmupHMC.reparam(Rep(PC(1.0), PC(0.0), Coordinate(2, 0.0), Coordinate(3, 0.0)), x[1], x) == closure
    @test WarmupHMC.reparam(Rep(PC(1.0), PC(0.0), CoordinateF(2, 0.0), CoordinateF(3, 0.0)), x[1], x) == closure
    constants = WarmupHMC.reparam(Rep(PC(1.0), PC(0.0), 0.5, log(2.0)), x[1], x)
    @test WarmupHMC.reparam(Rep(PC(1.0), PC(0.0), Coordinate(0, 0.5), Coordinate(0, log(2.0))), x[1], x) == constants
end

@testset "heterogeneous blocks with one accessor type: map, inverse, gradient" begin
    ir, irf = spec(Coordinate), spec(CoordinateF)
    @test isconcretetype(eltype(ir.pairs))
    ljac, y = ir(X)
    @test (ljac, y) == irf(X)
    @test y[FIXED] ≈ 0.5 .+ 2.0 .* X[FIXED]          # constant location 0.5, scale 2
    @test y[10:13] == X[10:13]                       # hyperparameters pass through
    ljac2, z = WarmupHMC.inverse(ir)(y)
    @test z ≈ X
    @test ljac + ljac2 ≈ 0 atol = 1e-12

    rp, rpf = RP(ir, ThreeBlocks(), AutoEnzyme()), RP(irf, ThreeBlocks(), AutoEnzyme())
    lp, g = LogDensityProblems.logdensity_and_gradient(rp, X)
    @test (lp, g) == LogDensityProblems.logdensity_and_gradient(rpf, X)
    @test lp ≈ ljac + LogDensityProblems.logdensity(ThreeBlocks(), y)
    @test g ≈ fd_gradient(x -> LogDensityProblems.logdensity(rp, x), X) rtol = 1e-5
end

# The concrete-element-type requirement stays: Julia widens a mixed vector and AD
# then fails late inside warm-up (IndexedReparametrization docstring). These are the
# reporter's two ways into it.
@testset "mixed accessor types across pairs are refused, naming the remedy" begin
    constants_beside_closures = () -> IR([
        [i => Rep(PC(1.0), PC(0.0), x -> x[10], x -> x[11]) for i in 1:3];
        [i => Rep(PC(1.0), PC(0.0), 0.5, log(2.0)) for i in FIXED]
    ])
    closures_from_two_sites = () -> IR([
        [i => Rep(PC(1.0), PC(0.0), x -> x[10], x -> x[11]) for i in 1:3];
        [i => Rep(PC(1.0), PC(0.0), x -> x[12], x -> x[13]) for i in 4:6]
    ])
    for build in (constants_beside_closures, closures_from_two_sites)
        err = try
            build()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("must have a concrete element type", err.msg)
        @test occursin("different places in the source are different types", err.msg)
        @test occursin("one callable struct", err.msg)
        # The concrete rule type, accessor tuple included (the widened eltype hides it).
        @test occursin(r"the first pair's rule is a \S*Reparametrization\{.*Tuple\{", err.msg)
    end
    # One comprehension site is one closure type, constants within a rule are fine.
    @test IR([i => Rep(PC(1.0), PC(0.0), x -> x[10 + 2(i > 3)], 0.0) for i in 1:6]) isa IR
end

# An argument that is neither a Real nor callable used to be accepted and fail at
# the first gradient; the scalar transform cannot use it either way.
@testset "an accessor that is neither a Real nor callable is refused at construction" begin
    for bad in (NotCallable(10), [0.5], :mu)
        err = try
            IR([i => Rep(PC(1.0), PC(0.0), bad, 0.0) for i in 1:3])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("argument 3 of the Reparametrization for index 1 is a $(typeof(bad)),", err.msg)
        @test occursin("neither a Real constant nor callable", err.msg)
    end
    err = try
        IR([1 => Rep(PC(1.0), PC(0.0), 0.0, NotCallable(11))])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("argument 4", err.msg)
    @test IR([]) isa IR
end

# The reporter's run failed during sampling. Adaptation writes new pairs back
# (`Reparametrization(value.target, new_source, value.args...)`), transports the
# pool through the old and new maps and back-transforms the draws, so the whole
# warm-up must treat the two accessor spellings identically: same seed, same draws.
@testset "adaptive warm-up with callable-struct accessors" begin
    @test !isnothing(Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt))
    config = (; n_draws=60, n_evaluations=32, stepsize_adaptation_limit=8,
        recording_target=32, max_tree_depth=5, variance_cond_target=1.0,
        nonlinear_adapt=true, init=(; position=copy(X), squared_scale=ones(13)))
    rp, rpf = RP(spec(Coordinate), ThreeBlocks(), AutoEnzyme()), RP(spec(CoordinateF), ThreeBlocks(), AutoEnzyme())
    result = adaptive_warmup_mcmc(Xoshiro(3), rp; config...)
    reference = adaptive_warmup_mcmc(Xoshiro(3), rpf; config...)
    @test size(result.posterior_position, 2) >= config.n_draws
    @test result.posterior_position == reference.posterior_position
    sources = [last(p).c for p in WarmupHMC.reparam_sources(rp)]
    @test sources == [last(p).c for p in WarmupHMC.reparam_sources(rpf)]
    @test any(!=(0.0), sources)                      # adaptation moved a centering
    @test all(p -> last(p).args isa NTuple{2,Coordinate}, WarmupHMC.reparametrizer(rp).pairs)
end
end
