module WarmupHMCLogDensityProblemsADExt

using WarmupHMC, LogDensityProblemsAD

import LogDensityProblemsAD: ADGradientWrapper

WarmupHMC.lja_reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw::AbstractVector) = begin 
    WarmupHMC.lja_reparametrize(parent(source), parent(target), draw)
end
WarmupHMC.find_reparametrization(source::ADGradientWrapper, draws; kwargs...) = begin
    WarmupHMC.reparametrize(source, WarmupHMC.find_reparametrization(parent(source), draws; kwargs...))
end
WarmupHMC.reparametrization_parameters(source::ADGradientWrapper) = WarmupHMC.reparametrization_parameters(
    parent(source)
) 

end