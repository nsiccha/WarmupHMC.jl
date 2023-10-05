module WarmupHMCLogDensityProblemsADExt

using WarmupHMC, LogDensityProblemsAD

import LogDensityProblemsAD: ADGradientWrapper

WarmupHMC.lja_reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw::NamedTuple, lja=0.) = begin 
    WarmupHMC.lja_reparametrize(parent(source), parent(target), draw, lja)
end

WarmupHMC.lpdf_and_invariants(source::ADGradientWrapper, draw::AbstractVector, lpdf=0.) = begin 
    WarmupHMC.lpdf_and_invariants(parent(source), draw, lpdf)
end
WarmupHMC.find_reparametrization(source::ADGradientWrapper, draws; kwargs...) = begin
    WarmupHMC.reparametrize(source, WarmupHMC.find_reparametrization(parent(source), draws; kwargs...))
end
WarmupHMC.reparametrization_parameters(source::ADGradientWrapper) = WarmupHMC.reparametrization_parameters(
    parent(source)
) 

end