# [Jet transport: the simple pendulum](@id pendulum)

In this example we illustrate the use of jet transport techniques
in `TaylorIntegration.jl` for the simple pendulum. We propagate a
neighborhood ``U_0`` around an initial condition ``q_0`` parametrized by the
sum ``q_0+\xi``, where ``q_0=(x_0,p_0)`` represents
the coordinates of the initial condition in phase space, and ``\xi=(\xi_1,\xi_2)``
represents an small variation with respect to this initial condition. We
re-interpret each component of the sum ``q_0+\xi`` as a multivariate polynomial
in the variables ``\xi_1`` and ``\xi_2``; below, the maximum
order of the multivariate polynomial is fixed at 8. We propagate these multivariate
polynomials in time using Taylor's method.

The simple pendulum is defined by the Hamiltonian ``H(x, p) = \frac{1}{2}p^2-\cos x``;
the corresponding equations of motion are given by
```math
\begin{eqnarray*}
\dot{x} &=& p, \\
\dot{p} &=& -\sin x.
\end{eqnarray*}
```

We integrate this problem for a neighborhood ``U_0`` around the initial
condition ``q_0 = (x(t_0), p(t_0)) = (x_0, p_0)``. For concreteness
we take ``p_0=0`` and choose ``x_0`` such
that the pendulum librates; that is, we will choose a numerical value for the
energy ``E=H(x_0,p_0)=-\cos x_0`` such that the pendulum's motion in phase space
is "below" (inside) the region bounded by the separatrix. In this
case, the libration period ``T`` of the pendulum is
```math
\begin{equation*}
T=\frac{4}{\sqrt{2}}\int_0^{x_0}\frac{dx}{\sqrt{\cos x_0-\cos x}},
\end{equation*}
```
which can be expressed in terms of the complete elliptic integral of the first kind, ``K``:
```math
\begin{equation*}
T=4K(\sin(x_0/2)).
\end{equation*}
```

The Hamiltonian for the simple pendulum is:
```@example pendulum
H(x) = 0.5x[2]^2-cos(x[1])
nothing # hide
```
The equations of motion are:
```@example pendulum
function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end
nothing # hide
```

We define the `TaylorN` variables necessary to perform the jet transport;
`varorder` represents the maximum order of expansion in the variations ``\xi``.
```@example pendulum
const varorder = 8
using TaylorIntegration
ξ = set_variables("ξ", numvars=2, order=varorder)
```
Note that `TaylorSeries.jl` is `@reexport`-ed internally by `TaylorIntegration.jl`.

The nominal initial condition is:
```@example pendulum
q0 = [1.3, 0.0]
```
The corresponding initial value of the energy is:
```@example pendulum
H0 = H(q0)
```
The parametrization of the neighborhood ``U_0`` is represented by
```@example pendulum
q0TN = q0 .+ ξ
```

To understand how the jet transport technique works, we shall evaluate the
Hamiltonian at ``q_0+\xi`` in order to obtain the 8-th order Taylor expansion of
the Hamiltonian with respect to the variations ``\xi``, around the initial
condition ``q_0``:
```@example pendulum
H(q0TN)
```
Note that the 0-th order term of the expression above is equal to the value
`H(q0)`, as expected.

Below, we set some parameters for the Taylor integration. We use a method
of `taylorinteg` which returns the solution at `t0`, `t0+integstep`,
`t0+2integstep`,...,`tmax`, where `t0` and `tmax` are the initial and final
times of integration, whereas `integstep` is a time interval
chosen by the user; we use
the variable `tv = t0:integstep:tmax` for this purpose and
choose `integstep` as ``\frac{T}{8}``.
```@example pendulum
order = 28     #the order of the Taylor expansion wrt time
abstol = 1e-20 #the absolute tolerance of the integration
using Elliptic # we use Elliptic.jl to evaluate the elliptic integral K
T = 4*Elliptic.K(sin(q0[1]/2)^2) #the libration period
t0 = 0.0        #the initial time
tmax = 6T       #the final time
integstep = T/8 #the time interval between successive evaluations of the solution vector
nothing # hide
```

We perform the Taylor integration using the initial condition `x0TN`, during
6 periods of the pendulum (i.e., ``6T``), exploiting multiple dispatch:
```@example pendulum
tv = t0:integstep:tmax # the times at which the solution will be evaluated
xv = taylorinteg(pendulum!, q0TN, tv, order, abstol)
nothing # hide
```

The integration above *works* for any initial neighborhood ``U_0``
around the nominal initial condition ``q_0``, provided it is sufficiently
small.

We will consider the particular case where ``U_0`` is a disk of radius ``r = 0.05``,
centered at ``q_0``; that is ``U_0=\{ q_0+\xi:\xi=(r\cos\phi,r\sin\phi); \phi\in[0,2\pi) \}``
for a given radius ``r>0``. We will denote by ``U_t`` the propagation of the
initial neighborhood ``U_0`` evaluated at time ``t``. Also, we denote by ``q(t)``
the coordinates of the nominal solution at time ``t``: ``q(t)=(x(t),p(t))``.
Likewise, we will denote the propagation at time ``t`` of a given initial
variation ``\xi_0`` by ``\xi(t)``. Then, we can compute the propagation of the
boundary ``\partial U_t`` of the neighborhood ``U_t``.
```@example pendulum
polar2cart(r, ϕ) = [r*cos(ϕ), r*sin(ϕ)] # convert radius r and angle ϕ to cartesian coordinates
r = 0.05 #the radius of the neighborhood
ϕ = 0.0:0.1:(2π+0.1) #the values of the angle
ξv = polar2cart.(r, ϕ)
nothing # hide
```

We evaluate the jet at ``\partial U_{x(t)}`` (the boundary of ``U_{x(t)}``) at each
value of the solution vector `xv`; we organize these values such that we can plot
them later:
```@example pendulum
xjet_plot = map(λ->λ.(ξv), xv[:,1])
pjet_plot = map(λ->λ.(ξv), xv[:,2])
nothing # hide
```
Above, we have exploited the fact that `Array{TaylorN{Float64}}` variables are
callable objects. Now, we evaluate the jet at the nominal solution, which
corresponds to ``\xi=(0,0)``, at each value of the solution vector `xv`:
```@example pendulum
x_nom = xv[:,1]()
p_nom = xv[:,2]()
nothing # hide
```

Finally, we shall plot the nominal solution (black dots), as well as the evolution of the
neighborhood ``U_0`` (in colors), each ``\frac{1}{8}``th of a period ``T``. The
initial condition corresponds to the black dot situated at ``q_0=(1.3,0)``
```@example pendulum
using Plots
plot( xjet_plot, pjet_plot,
    xaxis=("x",), yaxis=("p",),
    title="Simple pendulum phase space",
    leg=false, aspect_ratio=1.0
)
scatter!( x_nom, p_nom,
    color=:black,
    m=(1,2.8,stroke(0))
)
```
