# Fixed-kernel interruptible streaming sampler (`stream_mcmc` / `open_stream`).
#
# The core promise is "a process killed mid-sample or mid-write resumes from the
# immediately preceding state", so the interesting tests SIMULATE a crash
# deterministically — scribble a torn column past the safe index, or zero the
# freshest safe-ring slot — and assert the resume reproduces the exact draws an
# uninterrupted run would have produced. Byte-for-byte equality is the assertion,
# which requires single-threaded BLAS; the whole testset runs pinned to it and
# restores the previous setting afterwards.

problem = DiagGaussian([0.3, -1.0, 2.0, 0.0, 1.5], [0.5, 1.0, 2.0, 3.0, 0.25])
d = LogDensityProblems.dimension(problem)
q0 = zeros(d)
# metric = per-coordinate std devs (the ideal scale for this diagonal target).
kw = (; metric = problem.sigma, stepsize = 0.6, max_tree_depth = 8)

uninterrupted(seed, n; path) =
    Matrix(stream_mcmc(Xoshiro(seed), problem, q0; path, n_draws = n, kw...).draws)

_blas_threads = LinearAlgebra.BLAS.get_num_threads()
LinearAlgebra.BLAS.set_num_threads(1)
try
    @testset "fresh run: shape, finiteness, on-disk layout" begin
        p = tempname()
        r = stream_mcmc(Xoshiro(1), problem, q0; path = p, n_draws = 200, kw...)
        @test r.n_drawn == 200
        @test size(r.draws) == (d, 200)
        @test all(isfinite, r.draws)
        @test isfile(p) && isfile(p * ".ring")
        @test filesize(p) == d * 200 * sizeof(Float64)
        @test r.dimension == d
        @test r.position == Matrix(r.draws)[:, 200]   # last source-frame column
    end

    @testset "resume-only equals an uninterrupted run (byte-identical)" begin
        pRef = tempname()
        full = uninterrupted(42, 200; path = pRef)
        p = tempname()
        stream_mcmc(Xoshiro(42), problem, q0; path = p, n_draws = 100, kw...)   # partial
        r = stream_mcmc(problem; path = p, n_draws = 200, kw...)                 # resume
        @test r.n_drawn == 200
        @test Matrix(r.draws) == full
    end

    @testset "n_draws is a floor on resume" begin
        p = tempname()
        stream_mcmc(Xoshiro(3), problem, q0; path = p, n_draws = 120, kw...)
        r = stream_mcmc(problem; path = p, n_draws = 50, kw...)   # below written
        @test r.n_drawn == 120                                    # returns existing
    end

    @testset "crash recovery: a torn column past the safe index is recomputed" begin
        pRef = tempname()
        full = uninterrupted(7, 150; path = pRef)
        p = tempname()
        stream_mcmc(Xoshiro(7), problem, q0; path = p, n_draws = 40, kw...)      # safe at 40
        # Simulate a crash AFTER column 41 was (partially) written but BEFORE its
        # ring commit: scribble a NaN column at index 41, leaving the ring at 40.
        open(p, "r+") do io
            seek(io, 40 * d * sizeof(Float64))
            write(io, fill(NaN, d))
        end
        @test WarmupHMC._ring_read(p * ".ring")[3] == 40          # ring still says 40
        r = stream_mcmc(problem; path = p, n_draws = 150, kw...)
        @test r.n_drawn == 150
        @test all(isfinite, r.draws)                              # the NaN was overwritten
        @test Matrix(r.draws) == full                             # exact recovery
    end

    @testset "crash recovery: a torn ring slot falls back to the previous one" begin
        pRef = tempname()
        full = uninterrupted(9, 150; path = pRef)
        p = tempname()
        stream_mcmc(Xoshiro(9), problem, q0; path = p, n_draws = 60, kw...)
        ring = p * ".ring"
        seq, _, n_written, _, slot = WarmupHMC._ring_read(ring)
        cap = filesize(ring) ÷ 2
        open(ring, "r+") do io                                    # zero the freshest slot
            seek(io, slot * cap)
            write(io, zeros(UInt8, cap))
        end
        back = WarmupHMC._ring_read(ring)
        @test back !== nothing && back[3] == n_written - 1        # fell back one commit
        r = stream_mcmc(problem; path = p, n_draws = 150, kw...)
        @test r.n_drawn == 150
        @test Matrix(r.draws) == full                             # still exact
    end

    @testset "open_stream reads a run back; model==source for a plain density" begin
        p = tempname()
        full = Matrix(stream_mcmc(Xoshiro(11), problem, q0; path = p, n_draws = 90, kw...).draws)
        o = open_stream(p)
        @test o.n_drawn == 90
        @test Matrix(o.draws) == full
        @test o.position == full[:, 90]
        om = open_stream(p, problem; model = true)                # empty reparametrizer -> identity
        @test Matrix(om.draws) == full
    end

    @testset "overwrite vs. refusing to clobber an unreadable run" begin
        p = tempname()
        stream_mcmc(Xoshiro(1), problem, q0; path = p, n_draws = 40, kw...)
        r = stream_mcmc(Xoshiro(2), problem, q0; path = p, n_draws = 25, overwrite = true, kw...)
        @test r.n_drawn == 25                                     # fresh, discarded the old run
        # A samples file with no valid ring must not be silently clobbered.
        write(p * ".ring", zeros(UInt8, filesize(p * ".ring")))
        @test_throws ArgumentError stream_mcmc(Xoshiro(3), problem, q0; path = p, n_draws = 25, kw...)
        # ...and resume-only on an uncommitted ring errors rather than guessing.
        @test_throws ArgumentError stream_mcmc(problem; path = p, n_draws = 25, kw...)
    end

    @testset "progress reporting: exponential-window ESS + non-fixed fields" begin
        cap = CaptureProgress()
        p = tempname()
        r = stream_mcmc(Xoshiro(1), problem, q0; path = p, n_draws = 500,
            progress = _Treebars.ProgressNode(cap), description = "test", kw...)
        vals = [x isa Pair ? x.second : x for x in cap.seen]
        strs = filter(x -> x isa AbstractString, vals)
        ess_lines = unique(filter(s -> occursin("from", s) && occursin("draws", s), strs))
        @test 500 in filter(x -> x isa Integer, vals)             # bar reached the end
        @test any(s -> occursin("pending", s), strs)              # ESS pending sentinel first
        @test length(ess_lines) >= 3                              # doubling windows: 16,32,64,...,500
        @test any(s -> occursin("out of", s), strs)               # divergent (UncertainFrequency)
        @test any(s -> occursin("seconds", s), strs)              # draw-rate (Speed)
        @test r.ess isa AbstractVector && length(r.ess) == d && all(isfinite, r.ess)
        # The metric and step size are FIXED and must NOT be reported.
        @test !any(s -> occursin("stepsize", s), strs)
        # progress=nothing still fills a final ess in the return.
        r0 = stream_mcmc(Xoshiro(2), problem, q0; path = tempname(), n_draws = 60, kw...)
        @test all(isfinite, r0.ess)
    end

    @testset "seed from a WarmupHMC checkpoint, then resume the seeded stream" begin
        dir = mktempdir()
        adaptive_warmup_mcmc(Xoshiro(5), problem; checkpoint_dir = dir, n_draws = 40, progress = nothing)
        cp = joinpath(dir, "cp_latest.jls")
        @test isfile(cp)
        p = tempname()
        rs = stream_mcmc(cp, problem; path = p, n_draws = 80)     # kernel read from the checkpoint
        @test rs.n_drawn == 80
        @test all(isfinite, rs.draws)
        rs2 = stream_mcmc(cp, problem; path = p, n_draws = 160)   # resume the seeded run
        @test rs2.n_drawn == 160
        @test Matrix(rs2.draws)[:, 1:80] == Matrix(rs.draws)[:, 1:80]
    end

    @testset "checkpoint kernel + explicit start position (override)" begin
        dir = mktempdir()
        adaptive_warmup_mcmc(Xoshiro(5), problem; checkpoint_dir = dir, n_draws = 40, progress = nothing)
        cp = joinpath(dir, "cp_latest.jls")
        # Default start (payload's saved position) vs. an explicit override start:
        # warm-up moved off zeros, so the two chains differ.
        rdef = stream_mcmc(cp, problem; path = tempname(), n_draws = 50)
        rov  = stream_mcmc(cp, problem, q0; path = tempname(), n_draws = 50)
        @test rov.n_drawn == 50 && all(isfinite, rov.draws)
        @test Matrix(rdef.draws) != Matrix(rov.draws)
        # Same explicit position + same explicit rng ⇒ identical; a different rng ⇒ not.
        ra = stream_mcmc(cp, problem, q0; path = tempname(), n_draws = 50, rng = Xoshiro(123))
        rb = stream_mcmc(cp, problem, q0; path = tempname(), n_draws = 50, rng = Xoshiro(123))
        rc = stream_mcmc(cp, problem, q0; path = tempname(), n_draws = 50, rng = Xoshiro(999))
        @test Matrix(ra.draws) == Matrix(rb.draws)
        @test Matrix(rc.draws) != Matrix(ra.draws)
    end

    @testset "multi-chain (explicit kernel): per-chain files, chain == single-chain" begin
        dir = mktempdir()
        seeds  = [10, 20, 30]
        starts = [fill(0.1, d), fill(-0.2, d), fill(0.05, d)]
        rs = stream_mcmc([Xoshiro(s) for s in seeds], problem, starts;
            path = dir, n_draws = 120, parallel = true, kw...)
        @test rs isa AbstractVector && length(rs) == 3
        for (i, s) in enumerate(seeds)
            cp = joinpath(dir, "chain_$i")
            @test isfile(cp) && isfile(cp * ".ring")
            @test rs[i].n_drawn == 120
            # each chain reproduces the equivalent single-chain run bit-for-bit
            single = Matrix(stream_mcmc(Xoshiro(s), problem, starts[i];
                path = tempname(), n_draws = 120, kw...).draws)
            @test Matrix(rs[i].draws) == single
        end
    end

    @testset "multi-chain: serial and parallel agree per chain" begin
        starts = [fill(0.1, d), fill(-0.3, d)]
        par = stream_mcmc([Xoshiro(7), Xoshiro(8)], problem, starts;
            path = mktempdir(), n_draws = 90, parallel = true, kw...)
        ser = stream_mcmc([Xoshiro(7), Xoshiro(8)], problem, starts;
            path = mktempdir(), n_draws = 90, parallel = false, kw...)
        for i in 1:2
            @test Matrix(par[i].draws) == Matrix(ser[i].draws)
        end
    end

    @testset "multi-chain: per-chain resume equals an uninterrupted fan-out" begin
        starts = [fill(0.2, d), fill(-0.1, d), fill(0.0, d)]
        mk() = [Xoshiro(i) for i in 1:3]
        ref = stream_mcmc(mk(), problem, starts; path = mktempdir(), n_draws = 150, kw...)
        dir = mktempdir()
        stream_mcmc(mk(), problem, starts; path = dir, n_draws = 60, kw...)     # partial
        res = stream_mcmc(mk(), problem, starts; path = dir, n_draws = 150, kw...)  # resume
        for i in 1:3
            @test res[i].n_drawn == 150
            @test Matrix(res[i].draws) == Matrix(ref[i].draws)
        end
    end

    @testset "multi-chain from a checkpoint: derived rngs, independent, resumable" begin
        dir = mktempdir()
        adaptive_warmup_mcmc(Xoshiro(5), problem; checkpoint_dir = dir, n_draws = 40, progress = nothing)
        cp = joinpath(dir, "cp_latest.jls")
        starts = [fill(0.3, d), fill(-0.3, d), fill(0.0, d)]
        out = mktempdir()
        rs = stream_mcmc(cp, problem, starts; path = out, n_draws = 80)
        @test length(rs) == 3
        for i in 1:3
            @test rs[i].n_drawn == 80
            @test isfile(joinpath(out, "chain_$i"))
            @test all(isfinite, rs[i].draws)
        end
        @test Matrix(rs[1].draws) != Matrix(rs[2].draws)   # distinct starts + derived rngs
        rs2 = stream_mcmc(cp, problem, starts; path = out, n_draws = 160)   # resume the fan-out
        for i in 1:3
            @test rs2[i].n_drawn == 160
            @test Matrix(rs2[i].draws)[:, 1:80] == Matrix(rs[i].draws)[:, 1:80]
        end
    end
finally
    LinearAlgebra.BLAS.set_num_threads(_blas_threads)
end
