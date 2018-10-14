# [Poincaré map for the Hénon-Heiles system](@id rootfinding)

In this example, we shall illustrate how to construct a Poincaré map associated
with the surface of section ``y=0``, ``\dot y>0``, for ``E=0.1025`` for the
[Hénon-Heiles system](https://en.wikipedia.org/wiki/H%C3%A9non%E2%80%93Heiles_system). This is
equivalent to find the roots of an appropriate function `g(t,x,dx)`.

The Hénon-Heiles system is a 2-dof Hamiltonian system used to model the (planar)
motion of a star around a galactic center. The Hamiltonian is given by
``H = (p_x^2+p_y^2)/2 + (x^2+y^2)/2 + \lambda (x^2y-y^3/3)``, from which the
equations of motion can be obtained; below we concentrate in the case ``\lambda=1``.

```@example poincare
# Hamiltonian
V(x,y) = 0.5*( x^2 + y^2 )+( x*y^2 - x^3/3)
H(x,y,p,q) = 0.5*(p^2+q^2) + V(x, y)
H(x) = H(x...)

# Equations of motion
function henonheiles!(t, x, dx)
    dx[1] = x[3]
    dx[2] = x[4]
    dx[3] = -x[1]-(x[2]^2-x[1]^2)
    dx[4] = -x[2]-2x[2]*x[1]
    nothing
end
nothing # hide
```

We set the initial energy, which is a conserved quantity; `x0` corresponds
to the initial condition, which will be adjusted to be in the correct
energy surface.
```@example poincare
# initial energy and initial condition
const E0 = 0.1025
x0 = [0.45335, 0.0, 0.0, 0.0]
nothing # hide
```

In order to be able to generate (random) initial conditions with the appropriate
energy, we write a function `py`, which depends on `x`, `y`, `px` and
the energy `E` that returns the value of `py>0` for which the initial
condition `[x, y, px, py]` has energy `E`:
```@example poincare
# py: select py0 such that E=E0
py(x, E) = sqrt(2(E-V(x[1], x[2]))-x[3]^2)

# py!: in-place version of py; returns the initial condition
function py!(x, E)
    mypy = py(x, E)
    x[4] = mypy
    return x
end

# run py!
py!(x0, E0)
```

Let's check that the initial condition `x0` has actually energy equal to
`E0`, up to roundoff accuracy:
```@example poincare
H(x0)
```

The scalar function `g`, which may depend on the time `t`, the dependent
variable `x` and even the velocities `dx`, defines the surface of section by
means of the condition `g(t, x, dx)==0`. In the present case, it is defined as
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
nothing # hide
```
Note that in the definition of `g` we want to make sure that we only take the
"positive" crossings through the surface of section ``y=0``; hence the
`if...else...` block.

We initialize the auxiliary arrays where we shall save the solutions:
```@example poincare
# number of initial conditions
nconds = 100
tvSv = Vector{Vector{Float64}}(undef, nconds)
xvSv = Vector{Matrix{Float64}}(undef, nconds)
gvSv = Vector{Vector{Float64}}(undef, nconds)
x_ini = similar(x0)
nothing # hide
```

We generate `nconds` random initial conditions and integrate the equations of
motion from `t0=0` to `tmax=135`, using a polynomial of order 25 and absolute
tolerance `1e-25`:
```@example poincare
using TaylorIntegration

for i in 1:nconds
    rand1 = rand()
    rand2 = rand()
    x_ini .= x0 .+ 0.005 .* [sqrt(rand1)*cos(2pi*rand2), 0.0, sqrt(rand1)*sin(2pi*rand2), 0.0]
    py!(x_ini, E0)

    tv_i, xv_i, tvS_i, xvS_i, gvS_i = taylorinteg(henonheiles!, g, x_ini, 0.0, 135.0,
        25, 1e-25, maxsteps=30000);
    tvSv[i] = vcat(0.0, tvS_i)
    xvSv[i] = vcat(transpose(x_ini), xvS_i)
    gvSv[i] = vcat(0.0, gvS_i)
end
nothing # hide
```

We generate an animation with the solutions
```@example poincare
using Plots
poincareani5 = @animate for i=1:21
    scatter(map(x->x[i,1], xvSv), map(x->x[i,3], xvSv), label="$(i-1)-th iterate",
        m=(1,stroke(0)), ratio=:equal)
    xlims!(0.08, 0.48)
    ylims!(-0.13, 0.13)
    xlabel!("x")
    ylabel!("px")
    title!("Hénon-Heiles Poincaré map near a period 5 orbit")
end
gif(poincareani5, "henonheilespoincaremap5.gif", fps = 2)
nothing # hide
```

![Poincaré map for the Hénon Heiles system](henonheilespoincaremap5.gif)
