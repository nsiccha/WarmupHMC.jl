module DifferentiationInterfaceExt

using WarmupHMC, DifferentiationInterface, LogDensityProblems, LinearAlgebra

# The gradient half of `ReparametrizedProblem`; see that docstring for the
# user-facing contract, and `WarmupHMC._logdensity_and_gradient_reparam` for why
# it lives behind a weak dependency.
#
# Differentiate ONLY through the reparametrization transform (pure Julia) and
# reuse the inner problem's own `logdensity_and_gradient` — which may be an FFI
# call (BridgeStan) that no Julia AD backend could differentiate through anyway.
#
#     L(x)   = ljac(x) + ld(y(x))
#     ∂L/∂x  = ∂ljac/∂x + (∂y/∂x)' ∂ld/∂y
#
# `∂ld/∂y` comes from the inner problem; the rest comes from AD over the
# transform. The trick is that the whole right-hand side is the gradient of the
# SCALAR `x -> ljac(x) + dot(g_y, y(x))` with `g_y` frozen at its value at the
# current `y` — one reverse pass, not a full Jacobian. Freezing `g_y` is exactly
# what makes that identity hold: it is a constant of the differentiation, not a
# function of `x_`.
#
# Cost per gradient evaluation: one inner `logdensity_and_gradient`, one extra
# forward evaluation of the transform, and one AD pass over it. This is the
# gradient hot path, so the accessor closures inside each `Reparametrization`
# have to be AD-friendly.
function WarmupHMC._logdensity_and_gradient_reparam(p::WarmupHMC.ReparametrizedProblem, x::AbstractVector)
    ljac, y = p.reparametrizer(x)
    ld, g_y = LogDensityProblems.logdensity_and_gradient(p.problem, y)
    function reparam_objective(x_)
        ljac_, y_ = p.reparametrizer(x_)
        ljac_ + dot(g_y, y_)
    end
    _, g_x = value_and_gradient(reparam_objective, p.ad_backend, x)
    ljac + ld, g_x
end

end
