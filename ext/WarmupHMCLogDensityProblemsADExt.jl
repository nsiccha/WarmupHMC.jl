module WarmupHMCLogDensityProblemsADExt

using WarmupHMC, LogDensityProblemsAD

import LogDensityProblemsAD: ADGradientWrapper

# WarmupHMC.reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draws::AbstractMatrix) = begin 
#     WarmupHMC.reparametrize(parent(source), parent(target), draws)
# end

WarmupHMC.lja_reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw::AbstractVector) = begin 
    WarmupHMC.lja_reparametrize(parent(source), parent(target), draw)
end
# WarmupHMC.reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw::AbstractVector) = begin 
#     WarmupHMC.reparametrize(parent(source), parent(target), draw)
# end

# # WarmupHMC.lja(source::ADGradientWrapper, target::ADGradientWrapper, draws::AbstractMatrix) = begin 
# #     WarmupHMC.lja(parent(source), parent(target), draws)
# # end

# WarmupHMC.lja(source::ADGradientWrapper, target::ADGradientWrapper, draw::AbstractVector) = begin 
#     WarmupHMC.lja(parent(source), parent(target), draw)
# end

# WarmupHMC.find_reparametrization(source::ADGradientWrapper, draws::AbstractMatrix) = begin
#     WarmupHMC.reparametrize(source, WarmupHMC.find_reparametrization(parent(source), draws))
# end

end