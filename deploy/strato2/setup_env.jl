import Pkg

length(ARGS) == 1 || error("usage: setup_env.jl <WarmupHMC checkout>")

repo = abspath(only(ARGS))
env_dir = get(ENV, "WHMC_ENV_DIR", "/home/n/.local/share/WarmupHMC/env")
package_root = get(ENV, "WHMC_PACKAGE_ROOT", "")
isempty(package_root) && error("WHMC_PACKAGE_ROOT must name the materialized stack")

lock_path = joinpath(repo, "deploy", "strato2", "stack.lock")
isfile(lock_path) || error("missing stack lock at $lock_path")

packages = String[]
for line in eachline(lock_path)
    line = strip(first(split(line, '#'; limit=2)))
    isempty(line) && continue
    fields = split(line)
    length(fields) == 2 || error("bad stack.lock line: $line")
    package, want = fields
    path = joinpath(package_root, package)
    isdir(joinpath(path, ".git")) || error("missing package checkout $path")
    have = readchomp(`git -C $path rev-parse HEAD`)
    have == want || error("$package is at $have, expected $want")
    push!(packages, path)
end

append!(packages, [repo, joinpath(repo, "web")])
Pkg.activate(env_dir)
Pkg.develop(Pkg.PackageSpec[Pkg.PackageSpec(; path) for path in packages])
Pkg.add("Revise")
Pkg.instantiate()
Pkg.precompile()

println("WarmupHMC web environment ready at $env_dir")
