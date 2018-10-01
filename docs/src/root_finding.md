# [Root-finding: Poincaré map for the Hénon-Heiles system](@id rootfinding)

In this example, we will construct a Poincaré map associated to ``y=0``,
``\dot y>0`` and ``E=0.1025`` for the Hénon-Heiles system:
```@example poincare
# initial energy and initial condition
const E0, x02 = 0.1025, [0.45335, 0.0, 0.0, 0.0]

# the equations of motion for the Hénon-Heiles system
function henonheiles!(t, x, dx)
    dx[1] = x[3]
    dx[2] = x[4]
    dx[3] = -x[1]-(x[2]^2-x[1]^2)
    dx[4] = -x[2]-2x[2]*x[1]
    nothing
end
```
The potential and the Hamiltonian are:
```@example poincare
V(x,y) = 0.5*( x^2 + y^2 )+( x*y^2 - x^3/3)
H(x,y,p,q) = 0.5*(p^2+q^2) + VV(x, y)
H(x) = H(x...)
```
Since we will generate random initial conditions in order to construct the
Poincaré map, we will write a function `py`, which depends on `x`, `y`, `px` and
the energy `E`, and returns the value of `py` for which the initial
condition `[x, y, px, py]` has energy `E`:
```@example poincare
#select py0 such that E=E0
py(x,E) = sqrt(2(E-V(x[1],x[2]))-x[3]^2)
function py!(x,E)
    mypy = py(x,E)
    x[4] = mypy
    x
end
py!(x02,E0)
```
The scalar function `g`, which may depend on the time `t`, the dependent
variable `x` and even the velocities `dx` is:
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
# select initial energy and initial condition
const E0, x02 = 0.1025, [0.45335, 0.0, 0.0, 0.0]
# number of initial conditions
nconds2 = 100
@time for i in 1:nconds2
    rand1 = rand(); rand2 = rand()
    x_ini = py!(x02+0.005*[sqrt(rand1)*cos(2pi*rand2),0.0,sqrt(rand1)*sin(2pi*rand2),0.0],E0)
    tv_i, xv_i, tvS_i, xvS_i, gvS_i = taylorinteg(henonheiles!, g, x_ini, 0.0, 135.0, 25, 1e-25, maxsteps=30000);
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
