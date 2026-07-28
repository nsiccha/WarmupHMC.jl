@testitem "checkpoint writes are atomic" setup=[Determinism] tags=[:checkpoint] begin
    using WarmupHMC, Serialization
    using WarmupHMC: _atomic_serialize

    # A payload that blows up partway through serialization, to exercise the failure
    # path without having to kill a real process mid-write.
    struct _Explodes end
    Serialization.serialize(::AbstractSerializer, ::_Explodes) = error("boom")

    # Crash-safety guard for checkpoint writes.
    #
    # `_write_checkpoint` used to `serialize` straight to the final path. A crash
    # mid-write left a truncated `cp_latest.jls` — the file `resume_warmup_mcmc`
    # reads by default — so the checkpoint did not survive the one failure mode it
    # exists for. Writes now go to a temp file in the same directory and are renamed
    # into place, which is atomic within a filesystem.
    @testset "checkpoint writes are atomic" begin
        mktempdir() do dir
            path = joinpath(dir, "cp_latest.jls")

            # Round-trip, and overwrite-in-place.
            _atomic_serialize(path, (; a=1, b=[1.0, 2.0]))
            @test deserialize(path).a == 1
            _atomic_serialize(path, (; a=2, b=[3.0]))
            @test deserialize(path).a == 2

            # No temp litter left behind on the happy path.
            @test readdir(dir) == ["cp_latest.jls"]

            # THE POINT: a write that dies partway must leave the PREVIOUS checkpoint
            # intact and readable, not a truncated file.
            @test_throws Exception _atomic_serialize(path, (; a=3, boom=_Explodes()))
            @test deserialize(path).a == 2          # still the last good checkpoint
            @test readdir(dir) == ["cp_latest.jls"] # and no temp file stranded
        end
    end
end
