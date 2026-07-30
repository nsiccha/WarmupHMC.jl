# Unit tests for src/Reparametrizations.jl — the 4 exported reparametrization
# types (ReparametrizedProblem, IndexedReparametrization, PartiallyCentered,
# Reparametrization): round-trip / inverse correctness + logdensity wiring.

const PC = WarmupHMC.PartiallyCentered
const Rep = WarmupHMC.Reparametrization
const IR = WarmupHMC.IndexedReparametrization
const RP = WarmupHMC.ReparametrizedProblem
const reparam = WarmupHMC.reparam
const rinverse = WarmupHMC.inverse   # InverseFunctions.inverse, imported into WarmupHMC

@testset "PartiallyCentered reparam math" begin
    loc, log_scale = 1.3, -0.4
    x = 0.7
    # identity when target == source
    for c in (0.0, 0.5, 1.0)
        pc = PC(c)
        ljac, y = reparam(pc, pc, x, loc, log_scale)
        @test ljac ≈ 0
        @test y ≈ x
    end
    # non-centered (0) → centered (1):  y = loc + x*exp(log_scale),  ljac = log_scale
    ljac, y = reparam(PC(1.0), PC(0.0), x, loc, log_scale)
    @test ljac ≈ log_scale
    @test y ≈ loc + x * exp(log_scale)
    # centered (1) → non-centered (0):  y = (x-loc)*exp(-log_scale),  ljac = -log_scale
    ljac2, y2 = reparam(PC(0.0), PC(1.0), x, loc, log_scale)
    @test ljac2 ≈ -log_scale
    @test y2 ≈ (x - loc) * exp(-log_scale)
    # the two directions are mutually inverse
    ljac3, y3 = reparam(PC(1.0), PC(0.0), y2, loc, log_scale)
    @test y3 ≈ x
    @test ljac2 + ljac3 ≈ 0
end

@testset "Reparametrization inverse (constant args)" begin
    loc, log_scale = 0.5, 0.2
    r = Rep(PC(1.0), PC(0.0), loc, log_scale)
    x = 0.9
    ljac, y = reparam(r, x, nothing)             # 2nd arg is the vector for extractors; unused for constants
    @test (ljac, y) == reparam(PC(1.0), PC(0.0), x, loc, log_scale)
    ri = rinverse(r)
    @test ri.target.c == r.source.c              # inverse swaps target/source
    @test ri.source.c == r.target.c
    ljac2, z = reparam(ri, y, nothing)
    @test z ≈ x
    @test ljac + ljac2 ≈ 0
end

@testset "Reparametrization with extractor args" begin
    # loc from x[2], log_scale from x[3]
    r = Rep(PC(1.0), PC(0.0), x -> x[2], x -> x[3])
    xv = [0.9, 0.5, 0.2]                          # (value, loc, log_scale)
    ljac, y = reparam(r, xv[1], xv)
    @test ljac ≈ xv[3]
    @test y ≈ xv[2] + xv[1] * exp(xv[3])
end

@testset "IndexedReparametrization round-trip + passthrough" begin
    n = 5
    loc_idx, scale_idx = 4, 5                     # not among the transformed dims ⇒ passthrough
    ir = IR([
        i => Rep(PC(0.0), PC(1.0), x -> x[loc_idx], x -> x[scale_idx])
        for i in 1:3
    ])
    x = randn(n)
    ljac1, y = ir(x)
    @test y[loc_idx] == x[loc_idx]               # untransformed dims unchanged
    @test y[scale_idx] == x[scale_idx]
    @test y[1:3] != x[1:3]                        # transformed dims move
    ljac2, z = rinverse(ir)(y)                    # inverse recovers x
    @test z ≈ x
    @test ljac1 + ljac2 ≈ 0
    # empty ⇒ identity
    ir0 = IR([])
    lj, y0 = ir0(x)
    @test lj == 0
    @test y0 == x
end

@testset "ReparametrizedProblem logdensity" begin
    n = 5
    inner = DiagGaussian(zeros(n), ones(n))
    x = randn(n)
    # identity reparam ⇒ same dimension / capabilities / logdensity
    rp0 = RP(IR([]), inner)
    @test LogDensityProblems.dimension(rp0) == n
    @test LogDensityProblems.capabilities(typeof(rp0)) == LogDensityProblems.capabilities(typeof(inner))
    @test LogDensityProblems.logdensity(rp0, x) ≈ LogDensityProblems.logdensity(inner, x)
    # non-trivial reparam ⇒ logdensity = ljac + inner(y)
    loc_idx, scale_idx = 4, 5
    ir = IR([i => Rep(PC(0.0), PC(1.0), x -> x[loc_idx], x -> x[scale_idx]) for i in 1:3])
    rp = RP(ir, inner)
    ljac, y = ir(x)
    @test LogDensityProblems.logdensity(rp, x) ≈ ljac + LogDensityProblems.logdensity(inner, y)
end

# A coordinate sitting EXACTLY on its own centered location used to lose its whole
# gradient contribution. `reparam` computed `y` with `LogExpFunctions.xexpy(x -
# source.c * loc, ljac)`, and `xexpy` returns a CONSTANT zero when its first
# argument is zero and `ljac` is finite — a branch whose derivative is 0 where the
# true derivative is `exp(ljac)`. No error and no NaN, just a silently wrong
# gradient, so nothing here would have caught it except a finite-difference check
# aimed at the exact trigger.
#
# `source.c == 0` is the case to guard: fully non-centered is an ordinary
# configuration, and there the trigger collapses to `x_i == 0`, which is an
# entirely plausible coordinate value. Measured before the fix at
# `x = [0.7, 0, 0, 0, 0]`: 49% off central differences.
@testset "gradient at a coordinate exactly on its centered location" begin
    n = 5
    inner = DiagGaussian(zeros(n), ones(n))

    # `x_i - source.c * loc == 0` for every one of these, by construction
    triggers = [
        ("non-centered, one zero coordinate", 0.0, [0.7, 0.0, 0.3, 0.4, -0.1]),
        ("non-centered, several zeros",       0.0, [0.7, 0.0, 0.0, 0.0, -0.1]),
        # loc = x[4]; with source.c = 0.5 the trigger is x_i == 0.5 * x[4]
        ("half-centered, x_i == 0.5 * loc",   0.5, [0.4, 0.4, 0.4, 0.8, 0.25]),
    ]
    for (label, c_source, x) in triggers
        ir = IR([i => Rep(PC(1.0), PC(c_source), x -> x[4], x -> x[5]) for i in 1:3])
        rp = RP(ir, inner, AutoEnzyme())
        _, g = LogDensityProblems.logdensity_and_gradient(rp, x)
        ref = fd_gradient(z -> LogDensityProblems.logdensity(rp, z), x)
        @test g ≈ ref rtol = 1e-4
        # the specific failure was an exact zero where the reference is not zero
        for i in 1:3
            abs(ref[i]) > 1e-8 && @test !iszero(g[i])
        end
    end

    # the reparam formula itself: d/dx at x == source.c * loc must be exp(ljac),
    # not 0
    for (c_t, c_s, log_scale) in ((1.0, 0.0, 0.7), (1.0, 0.5, -0.4), (0.25, 1.0, 1.3))
        loc = 0.9
        x0 = c_s * loc                       # exactly the degenerate point
        ljac_expected = log_scale * (c_t - c_s)
        h = 1e-6
        _, yp = reparam(PC(c_t), PC(c_s), x0 + h, loc, log_scale)
        _, ym = reparam(PC(c_t), PC(c_s), x0 - h, loc, log_scale)
        @test (yp - ym) / 2h ≈ exp(ljac_expected) rtol = 1e-6
        # and the value at the point is still what the formula says
        ljac, y0 = reparam(PC(c_t), PC(c_s), x0, loc, log_scale)
        @test ljac ≈ ljac_expected
        @test y0 ≈ c_t * loc
    end
end
