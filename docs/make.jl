using Documenter
push!(LOAD_PATH,"src/")
using WarmupHMC

makedocs(
    sitename = "WarmupHMC",
    format = Documenter.HTML(),
    modules = [WarmupHMC]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nsiccha/WarmupHMC.jl.git"
)
