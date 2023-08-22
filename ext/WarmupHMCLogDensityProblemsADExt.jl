module WarmupHMCLogDensityProblemsADExt

using WarmupHMC, LogDensityProblemsAD

import LogDensityProblemsAD: ADGradientWrapper

WarmupHMC.reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw) = begin 
    WarmupHMC.reparametrize(parent(source), parent(target), draws)
end

WarmupHMC.lja(source::ADGradientWrapper, target::ADGradientWrapper, draw) = begin 
    WarmupHMC.lja(parent(source), parent(target), draws)
end

# WarmupHMC.find_reparametrization(source::ADGradientWrapper, draws::AbstractMatrix) = begin
#     WarmupHMC.reparametrize(source, WarmupHMC.find_reparametrization(parent(source), draws))
# end

end