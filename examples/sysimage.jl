cd("/home/niko/github/WarmupHMC.jl/examples")
using Pkg
Pkg.activate(".")
using PackageCompiler
using IJulia
using Revise

kernel_name = "warmup"
sysimage_path = joinpath(pwd(), "sysimage.so")

# create_sysimage(;
#     sysimage_path=sysimage_path,
#     # precompile_execution_file="code/julia/sysimage.jl"
# )
installkernel("julia-$kernel_name", "--project=@.", "--sysimage=$sysimage_path", "--threads", "8")