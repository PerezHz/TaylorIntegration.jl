# [Root-finding: Poincaré map for the Hénon-Heiles system](@id rootfinding)

In this example, we will construct a Poincaré map associated to ``y=0``,
``\dot y>0`` and ``E=0.1025`` for the Hénon-Heiles system:
```@example poincare
# initial energy and initial condition
const E0 = 0.1025
x0 = [0.45335, 0.0, 0.0, 0.0]

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
# py: select py0 such that E=E0
py(x, E) = sqrt(2(E-V(x[1], x[2]))-x[3]^2)
# py!: in-place version of py
function py!(x, E)
    mypy = py(x, E)
    x[4] = mypy
    x
end
# run py!
py!(x0, E0)
```
The scalar function `g`, which may depend on the time `t`, the dependent
variable `x` and even the velocities `dx` is:
```@example poincare
# y=0, py>0 section
function g(t, x, dx)
    # if py > 0...
    py_ = constant_term(x[4])
    if py_ > zero(py_)
        # ...return y
        return x[2]
    else
        #otherwise, return 0
        return zero(x[1])
    end
end
```
Note that in the definition of `g` we want to make sure that we only take the
"positive" crossings through the surface of section ``y=0``.
Auxiliary arrays where we will save the solutions:
```@example poincare
# number of initial conditions
nconds = 100
tvSv = Vector{Vector{Float64}}(undef, nconds)
xvSv = Vector{Matrix{Float64}}(undef, nconds)
gvSv = Vector{Vector{Float64}}(undef, nconds);
```
Generate random initial conditions and integrate:
```@example poincare
using TaylorIntegration
@time for i in 1:nconds
    rand1 = rand(); rand2 = rand()
    x_ini = py!(x0+0.005*[sqrt(rand1)*cos(2pi*rand2),0.0,sqrt(rand1)*sin(2pi*rand2),0.0],E0)
    tv_i, xv_i, tvS_i, xvS_i, gvS_i = taylorinteg(henonheiles!, g, x_ini, 0.0, 135.0, 25, 1e-25, maxsteps=30000);
    tvSv[i] = vcat(0.0, tvS_i)
    xvSv[i] = vcat(transpose(x_ini), xvS_i)
    gvSv[i] = vcat(0.0, gvS_i)
end
```
Plot the solution:
```@example poincare
using Plots
poincareani5 = @animate for i=1:21
    scatter(map(x->x[i,1], xvSv), map(x->x[i,3], xvSv), label="$(i-1)-th iterate", m=(1,stroke(0)), ratio=:equal)
    xlims!(0.08,0.48)
    ylims!(-0.13,0.13)
    xlabel!("x")
    ylabel!("pₓ")
    title!("Hénon-Heiles Poincaré map near a period 5 orbit")
end
gif(poincareani5, "henonheilespoincaremap5.gif", fps = 2) # hide
```

![Poincaré map for the Hénon Heiles system](henonheilespoincaremap5.gif)
