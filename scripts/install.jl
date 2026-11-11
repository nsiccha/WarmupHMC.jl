# Install this revision and its unregistered dependencies into the active
# consumer environment. Run with `julia --project=<your-env> scripts/install.jl`.
using Pkg, TOML

root = dirname(@__DIR__)
project = TOML.parsefile(joinpath(root, "Project.toml"))
specs = [PackageSpec(; url=source["url"], rev=source["rev"])
         for source in values(project["sources"]) if haskey(source, "url")]
revision = readchomp(`git -C $root rev-parse HEAD`)
push!(specs, PackageSpec(; url="https://github.com/nsiccha/WarmupHMC.jl", rev=revision))
Pkg.add(specs)
