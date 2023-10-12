module WarmupHMCOptimReverseDiffExt

using WarmupHMC, Optim, ReverseDiff

WarmupHMC.find_reparametrization(::Val{:ReverseDiff}, source, draws; iterations=16, method=LBFGS(), compiled=false) = begin 
    loss = WarmupHMC.reparametrization_loss_function(source, draws)
    init_arg = WarmupHMC.optimization_reparametrization_parameters(source)
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

WarmupHMC.find_reparametrization(::Val{:Optim}, source, draws; iterations=16, strict=false, kwargs...) = begin 
    init_arg = WarmupHMC.optimization_reparametrization_parameters(source)
    try
        if length(init_arg) == 0
            source
        elseif length(init_arg) == 1
            loss = WarmupHMC.reparametrization_loss_function(source, draws)
            optimization_result = optimize(
                loss, init_arg, 
                Optim.Options(iterations=iterations)
            )
            WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
        else
            WarmupHMC.find_reparametrization(:ReverseDiff, source, draws; iterations=iterations, kwargs...)
        end
    catch e
        @warn """
Failed to reparametrize ($iterations, $kwargs): 
$source
$(typeof(draws)), $(size(draws))
$(WarmupHMC.exception_to_string(e))
Not reparametrizing...
        """
        strict && rethrow(e)
        source
    end
end

end