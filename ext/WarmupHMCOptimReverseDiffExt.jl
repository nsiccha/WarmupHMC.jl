module WarmupHMCOptimReverseDiffExt

using WarmupHMC, Optim, ReverseDiff

WarmupHMC.find_reparametrization(::Val{:ReverseDiff}, source, draws; iterations=16, method=LBFGS(), compiled=false) = begin 
    loss = WarmupHMC.reparametrization_loss_function(source, draws)
    init_arg = WarmupHMC.reparametrization_parameters(source)
    loss_g! = if compiled 
        loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss, init_arg))
        (g, arg) -> ReverseDiff.gradient!(g, loss_tape, arg)
    else 
        (g, arg) -> ReverseDiff.gradient!(g, loss, arg)
    end
    try
        optimization_result = optimize(
            loss, loss_g!, init_arg, method, 
            Optim.Options(iterations=iterations)
        )
        @debug optimization_result
        WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
    catch e
        @warn """
Failed to reparametrize ($iterations, $method): 
    $source
    $draws
    $(WarmupHMC.exception_to_string(e))
Not reparametrizing...
        """
        source
    end
end

WarmupHMC.find_reparametrization(::Val{:Optim}, source, draws; iterations=16, kwargs...) = begin 
    init_arg = WarmupHMC.reparametrization_parameters(source)
    if length(init_arg) == 1
        loss = WarmupHMC.reparametrization_loss_function(source, draws)
        optimization_result = optimize(
            loss, init_arg, 
            Optim.Options(iterations=iterations)
        )
        WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
    else
        WarmupHMC.find_reparametrization(:ReverseDiff, source, draws; iterations=iterations, kwargs...)
    end
end

end