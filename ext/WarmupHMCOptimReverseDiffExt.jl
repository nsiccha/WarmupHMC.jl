module WarmupHMCOptimReverseDiffExt

using WarmupHMC, Optim, ReverseDiff

WarmupHMC.find_reparametrization(::Val{:ReverseDiff}, source, draws::AbstractMatrix; iterations=5, method=LBFGS(), compiled=false) = begin 
    loss = WarmupHMC.reparametrization_loss_function(source, draws)
    init_arg = WarmupHMC.reparametrization_parameters(source)
    loss_g! = if compiled 
        loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss, init_arg))
        (g, arg) -> ReverseDiff.gradient!(g, loss_tape, arg)
    else 
        (g, arg) -> ReverseDiff.gradient!(g, loss, arg)
    end
    optimization_result = optimize(
        loss, loss_g!, init_arg, method, 
        Optim.Options(iterations=iterations)
    )
    @debug optimization_result
    WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
end

end