# [Poincaré maps](@id rootfinding)

In this example, we shall illustrate how to construct a Poincaré map associated
with the surface of section ``y=0``, ``\dot y>0``, for ``E=0.1025`` for the
[Hénon-Heiles system](https://en.wikipedia.org/wiki/Hénon–Heiles_system). This is
equivalent to find the roots of an appropriate function `g(t, x, dx)`. We
illustrate the implementation using many initial conditions (Monte
Carlo like implementation), and then compare the results
with the use of [jet transport techniques](@ref jettransport).


## Monte Carlo simulation

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
to the initial condition, which will be properly adjusted to be in
the correct energy surface.
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
# py: select py0>0 such that E=E0
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

The scalar function `g`, which may depend on the time `t`, the vector of dependent
variables `x` and even the velocities `dx`, defines the surface of section by
means of the condition `g(t, x, dx) == 0`; `g` should return a variable of
type `eltype(x)`. In the present case, it is defined as
```@example poincare
# y=0, py>0 section
function g(t, x, dx)
    py_ = constant_term(x[4])
    # if py > 0...
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

!!! note
    Note that in the definition of `g` we want to make sure that we only take the
    "positive" crossings through the surface of section ``y=0``; hence the
    `if...else...` block.

We initialize some auxiliary arrays, where we shall save the solutions:
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
poincare_anim1 = @animate for i=1:21
    scatter(map(x->x[i,1], xvSv), map(x->x[i,3], xvSv), label="$(i-1)-th iterate",
        m=(1,stroke(0)), ratio=:equal)
    xlims!(0.08, 0.48)
    ylims!(-0.13, 0.13)
    xlabel!("x")
    ylabel!("px")
    title!("Hénon-Heiles Poincaré map (21 iterates)")
end
gif(poincare_anim1, "henonheilespoincaremap5.gif", fps = 2)
nothing # hide
```
![Poincaré map for the Hénon Heiles system](henonheilespoincaremap5.gif)


## [Jet transport](@id jettransport2)

Now, we illustrate the use of jet transport techniques in the same example,
that is, we propagate a neighborhood around `x0`, which will be plotted
in the Poincaré map. We first define the vector of small
increments of the phase space variables, `xTN`; we fix the maximum order
of the polynomial expansion in these variables to be `4`. Then,
`x0TN` is the neighborhood in the 4-dimensional phase space around ``x0``.
```@example poincare
xTN = set_variables("δx δy δpx δpy", numvars=length(x0), order=4)
x0TN = x0 .+ xTN
nothing # hide
```

As it was shown above, ``x0`` belongs to the energy surface
``H(x0) = E_0 = 0.1025``; yet, as it was defined above, the set of phase
space points denoted by `x0TN` includes points that belong to other
energy surfaces. This can be noticed by computing `H0(x0TN)`
```@example poincare
H(x0TN)
```
Clearly, the expression above may contain points whose energy is different from
`E0`. As it was done above, we shall fix the `py` component of `x0TN` so
*all* points of the neighborhood are in the same energy surface.
```@example poincare
py!(x0TN, E0) # Impose that all variations are on the proper energy shell!
H(x0TN)
```
We notice that the coefficients of all monomials whose order is not zero are
very small, and the constant_term is `E0`.

In order to properly handle this case, we need to extend the definition of
`g` to be useful for `Taylor1{TaylorN{T}}` vectors.
```@example poincare
#specialized method of g for Taylor1{TaylorN{T}}'s
function g(t, x::Array{Taylor1{TaylorN{T}},1}, dx::Array{Taylor1{TaylorN{T}},1}) where {T<:Number}
    py_ = constant_term(constant_term(x[4]))
    if py_ > zero( T )
        return x[2]
    else
        return zero(x[1])
    end
end
nothing # hide
```

We are now set to carry out the integration.
```@example poincare
tvTN, xvTN, tvSTN, xvSTN, gvSTN = taylorinteg(henonheiles!, g, x0TN, 0.0, 135.0, 25, 1e-25, maxsteps=30000);
nothing # hide
```

We define some auxiliary arrays, and then make an animation with the results for plotting.
```@example poincare
#some auxiliaries:
xvSTNaa = Array{Array{TaylorN{Float64},1}}(undef, length(tvSTN)+1 );
xvSTNaa[1] = x0TN
for ind in 2:length(tvSTN)+1
    whatever = xvSTN[ind-1,:]
    xvSTNaa[ind] = whatever
end
tvSTNaa = union([zero(tvSTN[1])], tvSTN);

myrnd  = 0:0.01:1
npoints = length(myrnd)
ncrosses = length(tvSTN)
xS = Array{Float64}(undef, ncrosses+1, npoints)
pS = Array{Float64}(undef, ncrosses+1, npoints)

myrad=0.005
ξx = @. myrad * cos(2pi*myrnd)
ξp = @. myrad * sin(2pi*myrnd)

for indpoint in 1:npoints
    xS[1,indpoint] = x0[1] + ξx[indpoint]
    pS[1,indpoint] = x0[3] + ξp[indpoint]
    mycond = [ξx[indpoint], 0.0, ξp[indpoint], 0.0]
    for indS in 2:ncrosses+1    
        temp = evaluate(xvSTNaa[indS], mycond)
        xS[indS,indpoint] = temp[1]
        pS[indS,indpoint] = temp[3]
    end
end

poincare_anim2 = @animate for i=1:21
    scatter(map(x->x[i,1], xvSv), map(x->x[i,3], xvSv), marker=(:circle, stroke(0)),
        markersize=0.01, label="Monte Carlo")
    plot!(xS[i,:], pS[i,:], width=0.1, label="Jet transport")
    xlims!(0.09,0.5)
    ylims!(-0.11,0.11)
    xlabel!("x")
    ylabel!("p")
    title!("Poincaré map: 4rd-order jet transport vs Monte Carlo")
end
gif(poincare_anim2, "poincareanim2.gif", fps = 2)
nothing # hide
```

![Poincaré map: Jet transport vs Monte Carlos](poincareanim2.gif)
