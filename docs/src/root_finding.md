# [Root-finding: Poincaré map for the Hénon-Heiles system](@id rootfinding)

In this example, we will construct a Poincaré map associated to ``y=0``,
``\dot y>0`` and ``E=0.1025`` for the Hénon-Heiles system.

The scalar function ``g``, which may depend on the time ``t``, the dependent
variable ``x`` and even the velocities ``dx`` is:
```@example poincare
g(t, x, dx) = x[2] # y=0 section
```
Auxiliary arrays where we will save the solution:
```@example poincare
tvSv2 = Array{Float64,1}[]
xvSv2 = Array{Float64,2}[]
gvSv2 = Array{Float64,1}[];
```
Generate random initial conditions and integrate:
```@example poincare
using TaylorIntegration
nconds2 = 100 #number of initial conditions
@time for i in 1:nconds2
    rand1 = rand(); rand2 = rand()
    x_ini = py!(x02+0.005*[sqrt(rand1)*cos(2pi*rand2),0.0,sqrt(rand1)*sin(2pi*rand2),0.0],E0)
    tv_i, xv_i, tvS_i, xvS_i, gvS_i = TaylorIntegration.poincare(henonheiles!, g, x_ini, 0.0, 135.0, 25, 1e-25, maxsteps=30000);
    push!(tvSv2, vcat(0.0, tvS_i))
    push!(xvSv2, vcat(transpose(x_ini), xvS_i))
    push!(gvSv2, vcat(0.0, gvS_i))
end
```
Plot the solution:
```@example poincare
using Plots
poincareani5 = @animate for i=1:21
    scatter(map(x->x[i,1], xvSv2), map(x->x[i,3], xvSv2), label="$(i-1)-th iterate", m=(1,stroke(0)), ratio=:equal)
    xlims!(0.08,0.48)
    ylims!(-0.13,0.13)
    xlabel!("x")
    ylabel!("pₓ")
    title!("Hénon-Heiles Poincaré map near a period 5 orbit")
end
gif(poincareani5, "./henonheilespoincaremap5.gif", fps = 2)
```
