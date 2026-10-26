@testitem "Pairwise Enzyme reparametrization gradient" setup=[WarmupHMCSharedFixtures] begin
    using Test, WarmupHMC
    using Random, LinearAlgebra
    using LogDensityProblems
    using DifferentiationInterface, Enzyme

# Snag find-reparametri-e2a5b2a7: under Enzyme the one-pass gradient of an
# IndexedReparametrization allocated for every accessor call. The extension
# differentiates pair by pair and must agree with the one-pass result to
# rounding. Only the order in which a coordinate read by several pairs sums their
# contributions may differ: by one ulp under runtime activity, and by 3e-14 on a
# correlated block under the default mode.

const PC = WarmupHMC.PartiallyCentered
const Rep = WarmupHMC.Reparametrization
const IR = WarmupHMC.IndexedReparametrization
one_pass(backend, ir, x, g) =
    invoke(WarmupHMC._reparam_gradient, Tuple{Any,Any,Any,Any}, backend, ir, x, g)
agrees(a, b) = isapprox(a, b; rtol = 1e-12)

# An accessor with a data-dependent loop.
struct RowSum <: Function
    first::Int
    last::Int
end
(a::RowSum)(x) = begin
    s = zero(eltype(x))
    for l in a.first:a.last
        s += tanh(x[l]) * x[l]
    end
    s
end
# One coordinate, as one concrete type, so pairs of different shapes share a vector.
struct At <: Function
    i::Int
end
(a::At)(x) = x[a.i]

# A correlated random-effect block: K terms per group, scales and a stick-breaking
# Cholesky factor shared by all groups, and each term's location recovered from the
# earlier terms' innovations. The accessors read the layout through one shared
# mutable state, as BRM's do. This is the shape that makes the one-pass gradient
# allocate for every accessor call (548 B per pair at K = 4); a loop, a tuple of
# innovations or an early return alone does not.
struct Block
    effects::Matrix{Int}
    cholesky_free::Vector{Int}
    log_scales::Vector{Int}
end
mutable struct BlockState{N}
    blocks::Vector{Block}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
end
struct BlockArg{KIND,N} <: Function
    state::BlockState{N}
    pair::Int
end
_tau(x, b::Block, i) = exp(x[b.log_scales[i]])
_cpc(x, b::Block, p) = tanh(x[b.cholesky_free[p]])
function _cholesky_entry(x, b, i, j)
    tau = _tau(x, b, i)
    stick = one(eltype(x))
    offset = (i - 1) * (i - 2) ÷ 2
    for l in 1:i-1
        z = _cpc(x, b, offset + l)
        l == j && return tau * (stick * z)
        stick *= sqrt(one(eltype(x)) - z * z)
    end
    j == i || throw(BoundsError((i, j)))
    tau * stick
end
function _block_location(x, state::BlockState{N}, p) where {N}
    bi, k, g = state.pair_blocks[p], state.pair_terms[p], state.pair_groups[p]
    T = eltype(x)
    k == 1 && return zero(T)
    block = state.blocks[bi]
    lookup = state.pair_lookup[bi]
    innovations = ntuple(_ -> zero(T), Val(N))
    m = zero(T)
    for i in 1:k
        m = zero(T)
        tau = _tau(x, block, i)
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _cpc(x, block, offset + l)
            m += (tau * (stick * z)) * innovations[l]
            stick *= sqrt(one(T) - z * z)
        end
        if i < k
            c = state.sources[lookup[i, g]]
            s = tau * stick
            innovations = Base.setindex(innovations, (x[block.effects[i, g]] - c * m) / s^c, i)
        end
    end
    m
end
(a::BlockArg{:location})(x) = _block_location(x, a.state, a.pair)
(a::BlockArg{:log_scale})(x) = begin
    s = a.state
    k = s.pair_terms[a.pair]
    log(_cholesky_entry(x, s.blocks[s.pair_blocks[a.pair]], k, k))
end
# The block's reparametrization and its dimension.
function correlated_block(K, G, rng)
    ncpc = K * (K - 1) ÷ 2
    effects = reshape(collect(1:K*G), K, G)
    block = Block(effects, collect(K*G + K .+ (1:ncpc)), collect(K*G .+ (1:K)))
    npairs = K * G
    terms = [k for g in 1:G for k in 1:K]
    groups = [g for g in 1:G for k in 1:K]
    lookup = zeros(Int, K, G)
    for p in 1:npairs
        lookup[terms[p], groups[p]] = p
    end
    state = BlockState{K - 1}([block], ones(Int, npairs), terms, groups, [lookup],
                              rand(rng, npairs))
    ir = IR([effects[terms[p], groups[p]] => Rep(PC(1.0), PC(state.sources[p]),
                                                 BlockArg{:location,K - 1}(state, p),
                                                 BlockArg{:log_scale,K - 1}(state, p))
             for p in 1:npairs])
    ir, K * G + K + ncpc
end

@test Base.get_extension(WarmupHMC, :WarmupHMCEnzymeExt) !== nothing

d = 40
rng = Xoshiro(17)
loop_pairs = [k => Rep(PC(1.0), PC(0.3 + 0.01k), RowSum(1, mod1(k, 5)), RowSum(1, 2))
              for k in 6:d]
cases = [
    "hierarchical closures" => (IR([k => Rep(PC(1.0), PC(0.4), x -> x[1], x -> x[2] / 2) for k in 3:d]), d),
    "constant location" => (IR([k => Rep(PC(0.7), PC(0.2), 0.5, x -> x[1]) for k in 2:d]), d),
    "loop accessors" => (IR(loop_pairs), d),
    # an accessor reads a coordinate another pair writes
    "reads a written coordinate" => (IR([k => Rep(PC(1.0), PC(0.1), x -> x[k - 1], x -> x[1]) for k in 3:d]), d),
    # a repeated index: only its last writer reaches y
    "repeated index" => (IR([[k => Rep(PC(1.0), PC(0.0), At(1), At(2)) for k in 3:d];
                             [5 => Rep(PC(0.5), PC(0.0), At(2), At(1))]]), d),
    "correlated block" => correlated_block(4, 10, rng),
]
backends = [
    "bare" => AutoEnzyme(),
    "Reverse" => AutoEnzyme(; mode = Enzyme.Reverse),
    "ReverseWithPrimal" => AutoEnzyme(; mode = Enzyme.ReverseWithPrimal),
    "runtime activity + Const" => AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                                             function_annotation = Enzyme.Const),
]

@testset "agrees with the one-pass gradient: $cname, $bname" for (cname, (ir, n)) in cases,
        (bname, backend) in backends
    for _ in 1:3
        x = 0.3 .* randn(rng, n)
        g = randn(rng, n)
        @test agrees(WarmupHMC._reparam_gradient(backend, ir, x, g), one_pass(backend, ir, x, g))
    end
end

@testset "logdensity_and_gradient takes the pairwise path" begin
    ir = first(last(cases[3]))
    inner = DiagGaussian(randn(rng, d), rand(rng, d) .+ 0.5)
    rp = WarmupHMC.ReparametrizedProblem(ir, inner, AutoEnzyme())
    x = 0.5 .* randn(rng, d)
    ljac, y = ir(x)
    ld, g_y = LogDensityProblems.logdensity_and_gradient(inner, y)
    value, gradient = LogDensityProblems.logdensity_and_gradient(rp, x)
    @test value == ljac + ld
    @test gradient == WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, g_y)
    @test agrees(gradient, one_pass(AutoEnzyme(), ir, x, g_y))
    @test gradient ≈ fd_gradient(q -> LogDensityProblems.logdensity(rp, q), x) rtol = 1e-6
end

@testset "forward mode keeps the one-pass method" begin
    ir = first(last(cases[1]))
    x = 0.5 .* randn(rng, d); g = randn(rng, d)
    @test WarmupHMC._reparam_gradient(AutoEnzyme(; mode = Enzyme.Forward), ir, x, g) ≈
          WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, g)
end

with_sources(ir, c) = IR([idx => Rep(value.target, PC(c), value.args...) for (idx, value) in ir.pairs])
one_pass_transport(backend, new_ir, old_ir, x, g_old) =
    invoke(WarmupHMC._transport_gradient_method, Tuple{Any,Any,Any,Any,Any},
           backend, new_ir, old_ir, x, g_old)

@testset "transport gradient agrees with the one-pass gradient: $cname, $bname" for (cname, (ir, n)) in cases,
        (bname, backend) in backends
    old_ir = with_sources(ir, 0.85)
    x = 0.3 .* randn(rng, n); g_old = randn(rng, n)
    pairwise! = WarmupHMC._transport_gradient_method(backend, ir, old_ir, x, g_old)
    one_pass! = one_pass_transport(backend, ir, old_ir, x, g_old)
    g1 = similar(x); g2 = similar(x)
    for _ in 1:3
        x .= 0.3 .* randn(rng, n); g_old .= randn(rng, n)
        pairwise!(g1, x, g_old)
        one_pass!(g2, x, g_old)
        @test agrees(g1, g2)
    end
end

@testset "allocation does not grow with the number of pairs" begin
    ir, n = correlated_block(4, 500, rng)
    x = 0.3 .* randn(rng, n); g = randn(rng, n)
    pairwise() = WarmupHMC._reparam_gradient(AutoEnzyme(), ir, x, g)
    pairwise(); one_pass(AutoEnzyme(), ir, x, g)
    # the gradient and the working seed, plus a constant
    @test @allocated(pairwise()) < 2 * 8n + 1024
    # positive control: on the same block the one-pass method allocates per pair
    @test @allocated(one_pass(AutoEnzyme(), ir, x, g)) > 4 * (2 * 8n + 1024)
    # and the sampler's gradient goes through the pairwise method: beyond it, only
    # the forward map and the inner problem's few vectors
    rp = WarmupHMC.ReparametrizedProblem(ir, DiagGaussian(zeros(n), ones(n)), AutoEnzyme())
    LogDensityProblems.logdensity_and_gradient(rp, x)
    @test @allocated(LogDensityProblems.logdensity_and_gradient(rp, x)) < 16 * 8n + 4096

    old_ir = with_sources(ir, 0.85)
    pairwise! = WarmupHMC._transport_gradient_method(AutoEnzyme(), ir, old_ir, x, g)
    one_pass! = one_pass_transport(AutoEnzyme(), ir, old_ir, x, g)
    g_new = similar(x)
    pairwise!(g_new, x, g); one_pass!(g_new, x, g)
    @test @allocated(pairwise!(g_new, x, g)) < 1024
    @test @allocated(one_pass!(g_new, x, g)) > 8n
end
end
