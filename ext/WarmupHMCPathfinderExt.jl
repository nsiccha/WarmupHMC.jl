module WarmupHMCPathfinderExt

using WarmupHMC, Pathfinder, UnPack
import WarmupHMC: TuningConfig, TuningState

TuningConfig{:pathfinder}(nruns::Int=1) = TuningConfig{:pathfinder}(;nruns)
TuningState(sampling_logdensity, tuning::TuningConfig{:pathfinder}, reparametrization_state) = begin 
    @unpack nruns = tuning
    @unpack rng, algorithm = sampling_logdensity
    @unpack reparametrization, warmup_state = reparametrization_state
    @unpack Q, κ, ϵ = warmup_state
    result_pf = if nruns in [1,-1] 
        rv = pathfinder(reparametrization; rng, ndraws=1, init=Q.q)
        if nruns == -1
            rv.draws[:, 1] .= mean(rv.fit_distribution)
        end
        rv
    else
        multipathfinder(reparametrization, 1; rng, nruns=nruns)
    end
    q = collect(result_pf.draws[:, 1])
    cov = result_pf.fit_distribution.Σ
    TuningState{:pathfinder}(;q, cov, ϵ, done=true)
end
end