# [Jet transport: the simple pendulum](@id pendulum)

In this example use show a simple example of the use of jet transport techniques
using `TaylorIntegration.jl` together with `TaylorSeries.jl`for the simple
pendulum, in order to propagate a neighborhood ``U_0`` around an initial
condition ``q_0``. The jet transport technique consists in considering the
neighborhood ``U_0`` as
being parametrized by the sum ``q_0+\xi``, where ``q_0=(x_0,p_0)`` represents
the coordinates of the initial condition in phase space, and ``\xi=(\xi_1,\xi_2)``
represents an small variation with respect to this initial condition. We
re-interpret each component of the sum ``q_0+\xi`` as a multivariate polynomial
in the variables ``\xi_1`` and ``\xi_2``. We propagate these multivariate
polynomials in time using Taylor's method.

The simple pendulum is defined by the Hamiltonian
```math
\begin{equation*}
H(x, p) = \frac{1}{2}p^2-\cos x.
\end{equation*}
```
The corresponding equations of motion are
```math
\begin{eqnarray*}
\dot{x} &=& p, \\
\dot{p} &=& -\sin x.
\end{eqnarray*}
```
We will integrate this problem for the neighborhood ``U_0`` around the initial
condition ``q_0 = (x(t_0), p(t_0)) = (x_0, p_0)``, using the jet transport
technique. For simplicity,
we will take ``p_0=0``. Furthermore, we will choose ``x_0`` such
that the pendulum librates; that is, we will choose a numerical value for the
energy ``E=H(x_0,p_0)=-\cos x_0`` such that the pendulum's motion in phase space
is "below" (inside) the region bounded by the separatrix. Then, for the initial
conditions given above, the librational period ``T`` of the pendulum is
```math
\begin{equation*}
T=\frac{4}{\sqrt{2}}\int_0^{x_0}\frac{dx}{\sqrt{\cos x_0-\cos x}}.
\end{equation*}
```
Or else, in terms of the complete elliptic integral of the first kind, ``K``:
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
We setup the `TaylorN` variables necessary to perform the jet transport;
`varorder` represents the order of expansion in the variations ``\xi``.
```@example pendulum
const varorder = 8
using TaylorIntegration
ξ = set_variables("ξ", numvars=2, order=varorder)
```
Note that we don't need to do `using TaylorSeries.jl`, since
`TaylorIntegration.jl` `@reexport`s variables and methods exported by
`TaylorSeries.jl`.

The nominal initial condition is:
```@example pendulum
q0 = [1.3, 0.0]
```
The initial value of the energy is:
```@example pendulum
H0 = H(q0)
```
The parametrization of the neighborhood ``U_0`` is represented by the following
array of `TaylorN`'s:
```@example pendulum
q0TN = q0 + ξ
```
To get a feel of how the jet transport technique works, we will evaluate the
Hamiltonian at ``q_0+\xi`` in order to obtain the 8-th order Taylor expansion of
the Hamiltonian wrt the variations ``\xi``, around the initial condition ``q_0``:
```@example pendulum
H(q0TN)
```
Note that the 0-th order term of the expression above is equal to the value
`H(q0)`, as expected.

Below, we setup some parameters for the Taylor integration. We will use a method
of `taylorinteg` which returns the solution only at `t0`, `t0+integstep`,
`t0+2integstep`,...,`tmax`, where `t0` and `tmax` are the initial and final
times of integration, whereas `integstep` is the time interval between
successive evaluations of the solution vector, chosen by the user; we will use
the variable `tv = t0:integstep:tmax` for this purpose. In this case, we are
choosing `integstep` to be ``\frac{1}{8}``-th of the period of the pendulum.
```@example pendulum
order = 28 #the order of the Taylor expansion wrt time
abstol = 1e-20 #the absolute tolerance of the integration
using Elliptic # we use Elliptic.jl to evaluate the elliptic integral K
T = 4*Elliptic.K(sin(q0[1]/2)^2) #the librational period
t0 = 0.0 #the initial time
tmax = 6T #the final time
integstep = T/8 #the time interval between successive evaluations of the solution vector
nothing # hide
```
Then, we perform a Taylor integration using the initial condition `x0TN`, during
6 periods of the pendulum (i.e., ``6T``), evaluating the solution each
`integstep = T/8`:
```@example pendulum
tv = t0:integstep:tmax # the times at which the solution will be evaluated
xv = taylorinteg(pendulum!, q0TN, t0:integstep:tmax, order, abstol)
nothing # hide
```
Note that we the integration above "works" for any initial neighborhood ``U_0``
around the nominal initial condition ``q_0``, of sufficiently small radius.

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
callable objects. Now, we will evaluate the jet at the nominal solution, which
corresponds to ``\xi=(0,0)``, at each value of the solution vector `xv`:
```@example pendulum
x_nom = xv[:,1]()
p_nom = xv[:,2]()
nothing # hide
```
Now, we plot the nominal solution (black dots), as well as the evolution of the neighborhood ``U_0`` (in colors), each ``\frac{1}{8}``th of a period. The initial condition corresponds to the black dot situated at ``q_0=(1.3,0)``
```@example pendulum
using Plots
plot(
    xjet_plot,
    pjet_plot,
    xaxis=("x",),
    yaxis=("p",),
    title="Simple pendulum phase space",
    leg=false,
    aspect_ratio=1.0
)
scatter!(
    x_nom,
    p_nom,
    color=:black,
    m=(1,2.8,stroke(0))
)
```
