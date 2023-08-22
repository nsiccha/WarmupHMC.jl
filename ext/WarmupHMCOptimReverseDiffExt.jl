module WarmupHMCOptimReverseDiffExt

using WarmupHMC, Optim, ReverseDiff

WarmupHMC.find_reparametrization(::Val{:ReverseDiff}, source, draws::AbstractMatrix; iterations=5, method=LBFGS(), verbose=false) = begin 
    loss = WarmupHMC.reparametrization_loss_function(source, draws)
    loss_g!(g, arg) = ReverseDiff.gradient!(g, loss_tape, arg)
    optimization_result = optimize(
        loss, loss_g!, WarmupHMC.reparametrization_parameters(source), method, 
        Optim.Options(iterations=iterations)
    )
    verbose && display(optimization_result)
    WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
end

end