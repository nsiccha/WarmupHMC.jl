using WarmupHMC, LogDensityProblems, LinearAlgebra, Random, Test
using WarmupHMC: cooperative_chain, clustered_chain

struct ExplicitInitLP
    dimension::Int
end
LogDensityProblems.dimension(p::ExplicitInitLP) = p.dimension
LogDensityProblems.capabilities(::Type{ExplicitInitLP}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::ExplicitInitLP, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::ExplicitInitLP, x) = (-sum(abs2, x) / 2, -x)

function captured_error(f)
    try
        f()
        nothing
    catch err
        err
    end
end

@testset "explicit NamedTuple initialization" begin
    dimension = 3
    lpdf = ExplicitInitLP(dimension)

    @testset "diagonal vector and matrix forms" begin
        squared_scales = (
            ones(dimension),
            fill(1e-3, dimension),
            Diagonal(ones(dimension)),
            Matrix(Diagonal(ones(dimension))),
            Matrix(1.0I, dimension, dimension) .+ 0.1,
        )
        for squared_scale in squared_scales
            result = adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension), squared_scale),
                n_draws=1,
                callback=(state, stage) -> stage === :init,
            )
            @test size(result.posterior_position) == (dimension, 0)
        end

        cooperative = cooperative_chain(
            Xoshiro(2), lpdf;
            init=(; position=zeros(dimension), squared_scale=ones(dimension)),
            n_draws=1,
            nonlinear_adapt=false,
        )
        @test diag(cooperative.scale_options.diagonal) == ones(dimension)

        clustered = clustered_chain(
            Xoshiro(3), lpdf;
            init=(; position=zeros(dimension), squared_scale=ones(dimension)),
            n_draws=1,
        )
        @test diag(clustered.scale) == ones(dimension)
    end

    @testset "actionable shape validation" begin
        err = captured_error() do
            adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension), squared_scale=ones(dimension - 1)),
                n_draws=1,
            )
        end
        @test err isa ArgumentError
        @test occursin("diagonal variance vector must have length 3", sprint(showerror, err))

        err = captured_error() do
            adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension), squared_scale=ones(dimension, dimension - 1)),
                n_draws=1,
            )
        end
        @test err isa ArgumentError
        @test occursin("full squared-scale matrix must have size (3, 3)", sprint(showerror, err))

        err = captured_error() do
            adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension - 1), squared_scale=ones(dimension)),
                n_draws=1,
            )
        end
        @test err isa ArgumentError
        @test occursin("init.position must have length 3", sprint(showerror, err))

        err = captured_error() do
            adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension), squared_scale=(1.0, 1.0, 1.0)),
                n_draws=1,
            )
        end
        @test err isa ArgumentError
        @test occursin("must be either a diagonal variance vector", sprint(showerror, err))

        err = captured_error() do
            adaptive_warmup_mcmc(
                Xoshiro(1), lpdf;
                init=(; position=zeros(dimension)),
                n_draws=1,
            )
        end
        @test err isa ArgumentError
        @test occursin("must contain `position` and `squared_scale`", sprint(showerror, err))
    end
end
