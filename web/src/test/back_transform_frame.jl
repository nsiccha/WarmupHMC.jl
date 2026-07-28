using WarmupHMC, Test, Random, LinearAlgebra, LogDensityProblems
using WarmupHMC: cooperative_chain, run_chain!, chain_result, chain_draws

include(joinpath(@__DIR__, "ad_backend.jl"))

# Which FRAME do the returned draws live in?
#
# `reparametrize_direction.jl` pins that `reparametrize!` maps the right WAY.
# This file pins that it is CALLED AT ALL — the finalizers used to gate it on
# `nonlinear_adapt`, so with `nonlinear_adapt=false` the sampler worked in the
# reparametrizer's source frame and handed those source-frame draws straight
# back, with nothing in the return value saying so.
#
# `nonlinear_adapt` gates whether the centering is FITTED. It must not gate
# whether the result is reported in the model's own parametrization, because the
# sampler works in the source frame either way. The consumer that gets hurt is
# the obvious one: a pinned-`c` CONTROL ARM compared against an adapted arm.
# Both return `posterior_position`, the control's is in a different frame, and
# nothing distinguishes them — so the comparison silently summarises two frames
# as one.
#
# WHY THE 1000: the probe's two frames are a full 1000 apart by construction, so
# "which frame is this" is answered by inspection and there is no threshold to
# pick. A source-frame draw is O(1); a model-frame draw is 1000 ± O(1).

# y[1] ~ N(0, 1) and y[2] ~ N(1000, 1), in MODEL coordinates.
struct _FrameProbe end
LogDensityProblems.dimension(::_FrameProbe) = 2
LogDensityProblems.capabilities(::Type{_FrameProbe}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::_FrameProbe, y) = -(y[1]^2 + (y[2] - 1000.0)^2) / 2
LogDensityProblems.logdensity_and_gradient(p::_FrameProbe, y) =
    (LogDensityProblems.logdensity(p, y), [-y[1], -(y[2] - 1000.0)])

# y[2] = 1000 + x[2]: a constant location, unit scale. The source frame is
# therefore standard normal in both coordinates — well conditioned, so nothing
# below depends on the sampler coping with awkward geometry.
frame_probe_rp() = ReparametrizedProblem(
    IndexedReparametrization([
        2 => Reparametrization(PartiallyCentered(1.0), PartiallyCentered(0.0), 1000.0, 0.0),
    ]),
    _FrameProbe(), AutoForwardDiff(),
)

@testset "finalizers report draws in the model frame regardless of nonlinear_adapt" begin

    @testset "the probe's two frames really are 1000 apart" begin
        # If this fails, every assertion below is vacuous — it would be
        # comparing a frame against itself.
        rp = frame_probe_rp()
        _, y = WarmupHMC.reparametrizer(rp)([0.25, -0.5])
        @test y[1] == 0.25
        @test y[2] == 999.5
    end

    cfg = (; n_draws=200, monitor_ess=false, progress=nothing)

    @testset "adaptive_warmup_mcmc (finalize_warmup!)" begin
        for nonlinear_adapt in (false, true)
            res = adaptive_warmup_mcmc(Xoshiro(7), frame_probe_rp();
                                       nonlinear_adapt, cfg...)
            d = res.posterior_position
            @test size(d, 2) > 0
            # Model frame. Source frame would put coordinate 2 at ~0.
            @test all(>(500), d[2, :])
            @test all(<(500), d[1, :])
        end
    end

    @testset "run_chain! and chain_result (cooperative)" begin
        for nonlinear_adapt in (false, true)
            chain = run_chain!(cooperative_chain(Xoshiro(7), frame_probe_rp();
                                                 nonlinear_adapt, cfg...))
            d = chain_draws(chain)
            @test size(d, 2) > 0
            @test all(>(500), d[2, :])

            # `chain_result` is the scheduler's finalizer and reparametrizes on
            # its own; drive a FRESH chain with `advance_window!` (which does
            # not) so this is not the forbidden double transform.
            c2 = cooperative_chain(Xoshiro(7), frame_probe_rp(); nonlinear_adapt, cfg...)
            while WarmupHMC.advance_window!(c2) != :done
            end
            @test all(>(500), chain_result(c2).posterior_position[2, :])
        end
    end

    @testset "still a no-op for an lpdf with no reparametrizer" begin
        # `reparametrizer(::Any)` is an empty `IndexedReparametrization` and
        # `reparametrize!` returns immediately on `isempty(ir.pairs)`, so
        # dropping the guard cannot perturb a plain problem. Byte-identity, not
        # approximate agreement — the whole safety argument for the change.
        a = adaptive_warmup_mcmc(Xoshiro(3), _FrameProbe(); nonlinear_adapt=false, cfg...)
        b = adaptive_warmup_mcmc(Xoshiro(3), _FrameProbe(); nonlinear_adapt=true, cfg...)
        @test a.posterior_position == b.posterior_position
        # And the plain target is genuinely sampled in its own coordinates, so
        # the 1000 shows up with no transform in sight.
        @test all(>(500), a.posterior_position[2, :])
    end
end
