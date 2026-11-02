@testitem "run JSON compatibility and atomic publication" begin
    using JSON

    fixtures = joinpath(@__DIR__, "fixtures", "run_json")
    inputs = include(joinpath(fixtures, "inputs.jl"))
    mktempdir() do dir
        for (name, fields) in pairs(inputs)
            path = joinpath(dir, "nested", string(name) * ".json")
            @test WarmupHMC._write_json(path, fields) === nothing
            actual = JSON.parsefile(path)
            @test actual == JSON.parsefile(joinpath(fixtures, string(name) * ".json"))
            @test actual["schema_version"] isa Int
        end
        @test WarmupHMC.checkpoint_schema_version() == 2
        @test length(readdir(joinpath(dir, "nested"))) == length(inputs)

        # Exercise the caller that folds completion exceptions into text and
        # symbols/optional chain diagnostics into the shared JSON writer.
        message = "α😀\tline\nreturn\rnull\0quote\"slash\\"
        completion = (;
            schema_version=3, policy=:completion_quorum, stuck_chain_indices=Int[],
            chains=[(; status=:failed, finished_at_seconds=1.25,
                n_samples=nothing, n_retained_samples=0, error=ErrorException(message),
                stuck_reason=nothing, stepsize=NaN, reference_stepsize=Inf,
                divergent_fraction=-Inf)],
        )
        WarmupHMC._write_completion_summary(dir, completion)
        summary = JSON.parsefile(joinpath(dir, "run_summary.json"))
        @test summary["policy"] == "completion_quorum"
        @test summary["chain_statuses"] == ["failed"]
        @test summary["chain_errors"] == [message]
        @test summary["chain_n_samples"] == [nothing]
        @test summary["chain_stepsizes"] == [nothing]
        @test summary["chain_reference_stepsizes"] == [nothing]
        @test summary["chain_divergent_fractions"] == [nothing]
    end

    struct UnwritableRunField end
    JSON.lower(::UnwritableRunField) = error("controlled serialization failure")

    mktempdir() do dir
        path = joinpath(dir, "run_summary.json")
        fields = Pair{String,Any}["field" => UnwritableRunField()]
        @test_throws "controlled serialization failure" WarmupHMC._write_json(path, fields)
        @test isempty(readdir(dir))

        write(path, "previous complete record")
        @test_throws "controlled serialization failure" WarmupHMC._write_json(path, fields)
        @test read(path, String) == "previous complete record"
        @test readdir(dir) == ["run_summary.json"]
    end
end
