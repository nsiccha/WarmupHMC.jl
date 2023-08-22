module WarmupHMCOptimReverseDiffext

using WarmupHMC, Optim, ReverseDiff

find_reparametrization(::Val{:ReverseDiff}, source, draws::AbstractMatrix; iterations=5, method=LBFGS(), verbose=false) = begin 
    loss = reparametrization_loss_function(source, draws)
    loss_g!(g, arg) = ReverseDiff.gradient!(g, loss_tape, arg)
    optimization_result = optimize(
        loss, loss_g!, reparametrization_parameters(source), method, 
        Optim.Options(iterations=iterations)
    )
    verbose && display(optimization_result)
    reparametrize(source, Optim.minimizer(optimization_result))
end

end