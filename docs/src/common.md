# [Using TaylorIntegration.jl from DifferentialEquations.jl](@id diffeqinterface)

Below, we show an example of usage of `TaylorIntegration.jl` from
`DifferentialEquations.jl`.

The problem we will integrate in this example is the Planar, Circular Restricted
Three-Body Problem (PCR3BP).


```@example common
using ParameterizedFunctions
# using LinearAlgebra: norm

f = @ode_def PCR3BP begin
    dx = px + y
    dy = py - x
    dpx = - (1-μ)*(x-μ)*((x-μ)^2+y^2)^-1.5 - μ*(x+1-μ)*((x+1-μ)^2+y^2)^-1.5 + py
    dpy = - (1-μ)*y    *((x-μ)^2+y^2)^-1.5 - μ*y      *((x+1-μ)^2+y^2)^-1.5 - px
end μ

μ = 0.01

V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
H(x, y, px, py) = (px^2+py^2)/2-(x*py-y*px) + V(x, y)
H(x) = H(x...)

u0 = u0 = [0.8, 0.0, 0.0, 1.0] # near-collision (with secondary) orbit
tspan = (0.0, 50.0)
p = [μ]
using TaylorIntegration
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, TaylorMethod(28), abstol=1e-20)

E0 = H(u0)
E = H.(sol.u);
δE = E .- E0;

# @show δE;
using Plots
plot(sol, vars=(1, 2))
```