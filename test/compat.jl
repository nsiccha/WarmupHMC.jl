# `Project.toml`'s [compat] table, checked against the [deps] table and against
# what the dependencies themselves require.
#
# WHY THIS FILE EXISTS
#
# Two defects were found by reading the file by hand, which is not a process:
#
# 1. `Treebars` — a hard runtime dependency, and the one blocking registration —
#    had NO compat entry at all. Four stdlibs had none either. A missing entry is
#    invisible: `Pkg` resolves fine without it, the suite stays green, and
#    General's AutoMerge is the first thing that ever says so.
#
# 2. `julia = "1.9"` was declared. That was not a stale bound, it was an
#    IMPOSSIBLE one — nine of the thirteen non-stdlib direct dependencies declare
#    `julia = "1.10"` themselves, so no 1.9 resolution of this project has ever
#    existed. The claim promised support that could not be installed, and it
#    survived because nothing compares our bound against theirs.
#
# Both are the same shape: a hand-maintained table that no consumer reads, so it
# cannot fail. The tests below make the [deps] table itself the source of truth,
# so a dependency added without a compat entry goes red here rather than at
# registration time.

# Lower bound of a compat string: "1.10" -> v1.10.0, "^1.6.7" -> v1.6.7,
# "0.6, 0.7" -> v0.6.0 (first range wins), "1" -> v1.0.0.
# Returns `nothing` for anything unparseable rather than guessing.
function compat_lower_bound(spec::AbstractString)
    first_range = strip(first(split(spec, ',')))
    cleaned = strip(first_range, ['^', '=', '~', '>', '<', ' '])
    isempty(cleaned) && return nothing
    parts = split(cleaned, '.')
    length(parts) == 1 && (parts = [parts[1], "0", "0"])
    length(parts) == 2 && (parts = [parts[1], parts[2], "0"])
    try
        VersionNumber(join(parts[1:3], '.'))
    catch
        nothing
    end
end

const PROJECT = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))

@testset "every dependency has a [compat] entry" begin
    deps = sort(collect(keys(PROJECT["deps"])))
    compat = keys(get(PROJECT, "compat", Dict{String,Any}()))

    # Guard the guard: if the [deps] table ever fails to parse, the loop below
    # passes vacuously and this becomes a test that cannot fail.
    @test !isempty(deps)

    for d in deps
        @test d in compat
    end
    @test "julia" in compat
end

@testset "Pathfinder compat includes the Turing 0.46 release" begin
    compat = Pkg.Types.semver_spec(PROJECT["compat"]["Pathfinder"])

    @test v"0.9.31" in compat
    @test !(v"0.10.6" in compat)
    @test v"0.10.7" in compat
    @test !(v"0.11.0" in compat)
end

@testset "declared julia bound is not below any dependency's" begin
    ours = compat_lower_bound(PROJECT["compat"]["julia"])
    @test ours !== nothing

    # Ask the RESOLVED dependencies what they require, rather than re-listing
    # them here — a second hand-maintained list would have the same defect this
    # file exists to catch.
    resolved = Pkg.dependencies()
    checked = 0
    for (name, uuid) in PROJECT["deps"]
        info = get(resolved, Base.UUID(uuid), nothing)
        info === nothing && continue
        info.source === nothing && continue
        f = joinpath(info.source, "Project.toml")
        isfile(f) || continue
        theirs_spec = get(get(TOML.parsefile(f), "compat", Dict{String,Any}()), "julia", nothing)
        theirs_spec === nothing && continue
        theirs = compat_lower_bound(theirs_spec)
        theirs === nothing && continue
        checked += 1
        # `ours >= theirs`, reported with the culprit named on failure.
        if !(ours >= theirs)
            @error "declared julia bound is below a dependency's" ours dep = name theirs
        end
        @test ours >= theirs
    end

    # Guard the guard: stdlibs carry no julia compat and every dep could be
    # skipped, leaving the loop above asserting nothing at all.
    @test checked > 0
end
