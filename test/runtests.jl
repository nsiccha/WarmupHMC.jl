module WarmupHMCTests
using Test, Random, WarmupHMC, TestModules
include("WarmupHMCTests.jl")
end

using TestModules
runtests!(WarmupHMCTests)
