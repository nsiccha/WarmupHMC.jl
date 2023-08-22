module WarmupHMCLogDensityProblemsADExt

using WarmupHMC, LogDensityProblemsAD

WarmupHMC.reparametrize(source::ADGradientWrapper, target::ADGradientWrapper, draw) = begin 
    WarmupHMC.reparametrize(parent(source), parent(target), draws)
end

end