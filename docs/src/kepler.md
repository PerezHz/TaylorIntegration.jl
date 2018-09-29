# [The Kepler problem](@id kepler_problem)

The [Kepler problem](https://en.wikipedia.org/wiki/Kepler_problem)
corresponds to the study of the motion of two bodies which are influenced
by their mutual gravitational attraction. In the center of mass and
relative coordinates, the problem is reduced to the motion of one body
of mass ``m = m_1 m_2 / M``, which we shall refer as particle below,
attracted gravitationally by another
located at rest at the origin of mass ``M=m_1+m_2``.

In cartesian coordinates, the equations of motion can be written as
```math
\begin{eqnarray*}
\dot{x} &=& v_x,\\
\dot{y} &=& v_y,\\
\dot{v}_x &=& - \frac{G M x}{(x^2 + y^2)^{3/2}},\\
\dot{v}_y &=& - \frac{G M y}{(x^2 + y^2)^{3/2}}.
\end{eqnarray*}
```
For concreteness, we fix ``\mu = G M = 1``. The coordinates ``x`` and ``y``
are the relative coordinates (to the center of mass) of the particle,
and ``v_x`` and ``v_y`` its velocity. The function `kepler_eqs!` *mutates*
the vectors corresponding to the LHS of the equations of motion.

```@example kepler
function kepler_eqs!(t, q, dq)
    dq[1] = q[3]
    dq[2] = q[4]
    rr = ( q[1]^2 + q[2]^2 )^(3/2)
    dq[3] = - q[1] / rr
    dq[4] = - q[2] / rr
end;
```

For suitable initial conditions (such that the total energy is negative)
the solutions are ellipses with one focus at the origin, which can be
parameterized in terms of its semi-major axis ``a`` and its eccentricity ``e``.
We set the initial conditions for the particle at periapse, which we locate
on the positive x-axis. Using the semimajor axis and the eccentricity, we
write them as
```math
\begin{eqnarray*}
x_0 & = & a (1-e),\\
y_0 & = & 0,\\
v_{x_0} & = & 0,\\
v_{y_0} & = & \frac{l_z}{x_0} = m \frac{\sqrt{\mu a (1-e^2)}}{x_0},
\end{eqnarray*}
```
where ``l_z`` is the angular momentum. We set the mass of the particle
``m=1``, the semi-major axis ``a=1`` and the eccentricity ``e=0.8``.
Kepler's third law defines the period of the motion as
``T= 2\pi a^{3/2}``.

```@example kepler
const mu = 1.0
const mass = 1.0
const aKep = 1.0
const eKep = 0.8;
```

The initial conditions are then set using `ini_cond`
```@example kepler
function ini_cond(a, e)
    x0  = a*(one(e)-e)
    vy0 = mass * sqrt( mu * a * (1-e^2) ) / x0
    y0  = zero(vy0)
    vx0 = zero(vy0)
    return [x0, y0, vx0, vy0]
end
q0 = ini_cond(aKep, eKep)
```

We now perform the integration, using a 28 order expansion and
absolute tolerance of ``10^{-20}``.
```@example kepler
using TaylorIntegration, Plots
t, q = taylorinteg(kepler_eqs!, q0, 0.0, 10000*2pi, 28, 1.0e-20, maxsteps=700000);
t[end], q[end,:]
```

We first plot the orbit.
```@example kepler
x = view(q, :, 1)
y = view(q, :, 2)
vx = view(q, :, 3)
vy = view(q, :, 4)
plot(x, y, legend=false)
scatter!([0], [0], shape=:circle, ms=5)
xaxis!("x", (-2.0, 0.5))
yaxis!("y", (-1.0, 1.0))
title!("Fig. 1")
```

The following functions allow us to calculate the energy and angular
momentum using cartesian coordinates.

```@example kepler
function energy( x, y, vx, vy )
    kinetic = 0.5 * (vx*vx + vy*vy)
    r = sqrt( x*x + y*y)
    potential = - mu * mass / r
    return kinetic + potential
end
lz( x, y, vx, vy ) = mass * ( x*vy - y*vx ) ;
```

We use the change in energy and angular momentum of the orbit
with respect to the initial value of the corresponding quantity
as a function of time. These quantities are expressed
in units of the *local epsilon* of the initial
energy or angular momentum, respectively. This serves to illustrate
the accuracy of the calculation, shown in Figure 2 and 3.
```@example kepler
e0 = energy(q0...)
δE = (energy.(x,y,vx,vy) .- e0) ./ eps(e0)
plot(t, δE)
xlabel!("t")
ylabel!("dE")
title!("Fig. 2")
```

```@example kepler
lz0 = lz(q0...)
δlz = (lz.(x,y,vx,vy) .- lz0) ./ eps(lz0)
plot(t, δlz)
xlabel!("t")
ylabel!("dlz")
title!("Fig. 3")
```

These errors are reminiscent of random walks.

The maximum *absolute* errors of the energy and angular momentum
are
```@example kepler
maximum( abs.(energy.(x,y,vx,vy) .- e0) ), maximum( abs.(lz.(x,y,vx,vy) .- lz0) )
```
