module DifferentiationInterfaceExt

using WarmupHMC, DifferentiationInterface, LogDensityProblems, LinearAlgebra

function WarmupHMC._logdensity_and_gradient_reparam(p::WarmupHMC.ReparametrizedProblem, x::AbstractVector)
    # Differentiate only through the reparametrization transform (pure Julia),
    # using the inner problem's native logdensity_and_gradient (e.g. BridgeStan FFI).
    #
    # L(x) = ljac(x) + ld(y(x))
    # ∂L/∂x = ∂ljac/∂x + (∂y/∂x)^T ∂ld/∂y
    #
    # We compute ∂y/∂x and ∂ljac/∂x via AD through the reparametrization,
    # and ∂ld/∂y from the inner problem.

    # Forward: get y and ljac
    ljac, y = p.reparametrizer(x)

    # Inner problem gradient at y (uses native gradient, e.g. BridgeStan)
    ld, g_y = LogDensityProblems.logdensity_and_gradient(p.problem, y)

    # AD through the reparametrization to get Jacobian-vector product:
    # We need ∂(ljac + g_y' * y) / ∂x, which gives ∂ljac/∂x + (∂y/∂x)^T g_y
    # This is a scalar function of x, so we can use value_and_gradient.
    function reparam_objective(x_)
        ljac_, y_ = p.reparametrizer(x_)
        ljac_ + dot(g_y, y_)
    end
    _, g_x = value_and_gradient(reparam_objective, p.ad_backend, x)

    ljac + ld, g_x
end

end
