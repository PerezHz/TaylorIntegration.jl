# [Interoperability between TaylorIntegration.jl and the `JuliaDiffEq` package ecosystem](@id diffeqinterface)

Here, we show an example of interoperability between `TaylorIntegration.jl` and
some packages from the `JuliaDiffEqs` organization.

Below, we use `ParameterizedFunctions.jl` to define the appropriate system of ODEs.
Also, we use `OrdinaryDiffEq.jl`, in order to compare
the performance and accuracy of `TaylorIntegration.jl` with respect to
high-accuracy methods for non-stiff problems.

The problem we will integrate in this example is the Planar, Circular Restricted
Three-Body Problem (PCR3BP; also capitalized as PCRTBP). The PCR3BP describes
the motion of a body with mass $m_3$ under the gravitational influence of two
bodies with masses $m_1$ and $m_2$ such that $m_1>m_2$ and $m_3$ is much smaller
than the other two masses, and therefore the third body is modeled as a test particle.
The body with the greater mass $m_1$ is called the *primary*, whereas the body
with the lesser mass $m_2$ is called the *secondary*; these bodies are together
called the *primaries* and are assumed
to describe circular orbits about their center of mass, which is placed at
the origin of the reference frame. Further, the orbit of the third body is
assumed to take place in the orbital plane of the primary and the secondary.
A full treatment of the PCR3BP may be found in [[1]](@ref refsPCR3BP).

The quotient $\mu = m_2/(m_1+m_2)$ is known as the *mass parameter*. If we select
mass units such that $m_1+m_2=1$, then we have $m_1=1-\mu$ and $m_2=\mu$. In
this example, we assume the mass parameter to have a value $\mu=0.01$.
```@example common
μ = 0.01
nothing # hide
```
The Hamiltonian for the PCR3BP in the synodic frame (i.e., a frame which rotates
such that the primaries are at rest) is
```math
\begin{equation}
\label{eq-pcr3bp-hamiltonian}
H(x, y, p_x, p_y) = \frac{1}{2}(p_x^2+p_y^2) - (x p_y - y p_x) + V(x, y)
\end{equation}
```
where
```math
\begin{equation}
\label{eq-pcr3bp-potential}
V(x, y) = - \frac{1-\mu}{\sqrt{(x-\mu)^2+y^2}} - \frac{\mu}{\sqrt{(x+1-\mu)^2+y^2}}.
\end{equation}
```
is the gravitational potential associated to the primaries. The RHS of Eq.
(\ref{eq-pcr3bp-hamiltonian}) is also known as the *Jacobi constant*, since it is a
preserved quantity of motion in the PCR3BP. We will use this property to check
the accuracy of the obtained solution.
```@example common
V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
H(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)
H(x) = H(x...)
nothing # hide
```
The equations of motion for the PCR3BP are
```math
\begin{eqnarray}
\label{eqs-motion-pcr3bp}
    \dot{x} &=& p_x + y \\
    \dot{y} &=& p_y - x \\
    \dot{p_x} &=& - \frac{(1-\mu)(x-\mu)}{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu(x+1-\mu)}{((x+1-\mu)^2+y^2)^{3/2}} + p_y \\
    \dot{p_y} &=& - \frac{(1-\mu)y      }{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu y       }{((x+1-\mu)^2+y^2)^{3/2}} - p_x.
\end{eqnarray}
```
We define this system of ODEs with `ParameterizedFunctions.jl`
```@example common
using ParameterizedFunctions
f = @ode_def PCR3BP begin
    dx = px + y
    dy = py - x
    dpx = - (1-μ)*(x-μ)*((x-μ)^2+y^2)^-1.5 - μ*(x+1-μ)*((x+1-μ)^2+y^2)^-1.5 + py
    dpy = - (1-μ)*y    *((x-μ)^2+y^2)^-1.5 - μ*y      *((x+1-μ)^2+y^2)^-1.5 - px
end μ
nothing # hide
```
We will select initial conditions $q_0 = (x_0, y_0, p_{x,0}, p_{y,0})$ such that
$H(q_0) = J_0$, where $J_0$ is a prescribed value. In order to do this,
we select $y_0 = p_{x,0} = 0$ and compute the value of $p_{y,0}$ for which
$H(q_0) = J_0$ holds for a prescribed value $J_0$.

We want to select an adequate value for $J_0$ such that the test particle is
able to have close encounters with both primaries but cannot escape to infinity.
We may obtain a first
approximation to the desired value of $J_0$ if we plot the projection of the
zero-velocity curves on the $x$-axis.
```@example common
ZVC(x) =  -x^2/2 + V(x, zero(x)) # projection of the zero-velocity curves on the x-axis
using Plots
plot(ZVC, -2:0.001:2, label="zero-vel. curve")
plot!([-2.5, 2.5], [-1.5772, -1.5772], label="J0 = -1.5722")
ylims!(-3, -1)
title!("Zero-velocity curves (x-axis projection)")
```
Note that the maxima in the plot correspond to the Lagrangian points $L_1$, $L_2$
and $L_3$. Further, from the plot we see that an adequate value for $J_0$ is
$J_0 = -1.5722$.
```@example common
J0 = -1.5772
nothing # hide
```
Next, we define a function `py!`, which depends on the initial condition $q_0 = (x_0, 0, 0, p_{y,0})$
and the Jacobi constant value $J_0$, such that it computes an adequate value
$p_{y,0}$ for which we have $H(q_0)=J_0$ and updates the initial condition
accordingly.
```@example common
function py!(q0, J0)
    @assert q0[2] == zero(eltype(q0)) "q0[2] has to be equal to zero"
    @assert q0[3] == zero(eltype(q0)) "q0[3] has to be equal to zero"
    q0[4] = q0[1] + sqrt( q0[1]^2-2( V(q0[1], q0[2])-J0 ) )
    nothing
end
nothing # hide
```
We are now ready to generate an appropriate initial condition.
```@example common
q0 = [-0.8, 0.0, 0.0, 0.0]
py!(q0, J0)
q0
```
Note that the value of `q0` has been updated. We check that the value of the
Hamiltonian evaluated at the initial condition is indeed equal to `J0`.
```@example common
H(q0) == J0
```
We define an `ODEProblem` in order to integrate the problem with `TaylorIntegration.jl`
via its common interface bindings with `JuliaDiffEq`.
```@example common
tspan = (0.0, 1000.0)
p = [μ]
using TaylorIntegration
prob = ODEProblem(f, q0, tspan, p)
```
Solve `prob` using a 25-th order Taylor method, with a local absolute tolerance $\epsilon_\mathrm{tol} = 10^{-20}$.
```@example common
@time solT = solve(prob, TaylorMethod(25), abstol=1e-20);
```
We load `OrdinaryDiffEq` in order to solve the same problem `prob`
with the `Vern8` method, which the `DifferentialEquations.jl` [documentation](http://docs.juliadiffeq.org/stable/solvers/ode_solve.html#Non-Stiff-Problems-1)
recommends for high-accuracy (i.e., very low tolerance) integrations of
non-stiff problems.
```@example common
using OrdinaryDiffEq
solV = solve(prob, Vern8()) #solve `prob` with the `Vern8` method
nothing # hide
```
We plot the $x--y$ orbit from the solution obtained with `TaylorIntegration.jl`:
```@example common
plot(solT, vars=(1, 2))
scatter!([μ, -1+μ], [0,0], leg=false)
xlims!(-1+μ-0.2, 1+μ+0.2)
```
Note that the orbit obtained with `TaylorIntegration.jl` displays the expected
dynamics: the test particle is able to explore the regions sorrounding both
primaries, without escaping to infinity. As a comparison, we now plot the $x--y$
orbit from the solution obtained with the `Vern8()` method:
```@example common
plot(solV, vars=(1, 2))
scatter!([μ, -1+μ], [0,0], leg=false)
# xlims!(-1+μ-0.2, 1+μ+0.2)
```
In the `Vern8()` case, the displayed dynamics in the $x--y$ plane are not quite
what we should expect qualitatively from how we constructed the problem and the
initial conditions.

We can obtain a quantitative comparison of the performance of both integrations
if we check the preservation of the Jacobi constant:
```@example common
ET = H.(solT.u)
EV = H.(solV.u)
δET = ET .- J0
δEV = EV .- J0
nothing # hide
```
Below, we plot, in log scale, the `abs` of the absolute error in the Jacobi
constant as a function of time, for both the `TaylorIntegration.jl` and the
`Vern8()` solutions:
```@example common
plot(solT.t, abs.(δET), yscale=:log10)
plot!(solV.t, abs.(δEV))
ylims!(10^-18,10^4)
```
We see that, whereas the Jacobi constant error for the `TaylorIntegration.jl`
solution remains bounded below $10^{-13}$, the solution with `Vern8()` is larger
than $10^{-1}$. Hence, in this case, the `Vern8` does not reproduce the dynamics
of the PCR3BP for the selected initial conditions, which should both preserve
the Jacobi constant while being able to explore the regions sorrounding the
primaries without escaping to infinity. On the other hand, the
`TaylorIntegration.jl` solution reproduces the dynamics to a high accuracy level
even after repeated close encounters with the primaries.

### [References](@id refsPCR3BP)

[1] Murray, Carl D., Stanley F. Dermott. Solar System dynamics. Cambridge University Press, 1999.
