# [Interoperability with `DifferentialEquations.jl`](@id diffeqinterface)

Here, we show an example of interoperability between `TaylorIntegration.jl` and
[`DifferentialEquations.jl`](https://github.com/JuliaDiffEq/DifferentialEquations.jl), i.e.,
how to use `TaylorIntegration.jl` from the `DifferentialEquations`
ecosystem. The basic requirement is to load `DiffEqBase.jl`, which
sets-up the common interface.
Below, we shall also use `OrdinaryDiffEq.jl` to compare
the accuracy of `TaylorIntegration.jl` with respect to
high-accuracy methods for non-stiff problems (`Vern9` method).
While `DifferentialEquations` offers many macros to simplify certain
aspects, we do not rely on them simply because using properly `@taylorize`
improves the performance.

!!! note
    Currently, the only keyword arguments supported by `DiffEqBase.solve` that
    are implemented in `TaylorIntegration.jl` are `:saveat` and `:tstops`. There
    is also experimental support for `:callback`, both discrete and continuous;
    some examples may be found in `test/common.jl`. The keyword argument
    `:parse_eqs` is available in order to control the use of methods defined
    via [`@taylorize`](@ref).

The problem we will integrate in this example is the planar circular restricted
three-body problem (PCR3BP, also capitalized as PCRTBP). The PCR3BP describes
the motion of a body with negligible mass ``m_3`` under the gravitational influence of two
bodies with masses ``m_1`` and ``m_2``, such that ``m_1 \ge m_2``. It is
assumed that ``m_3`` is much smaller than the other two masses so it
does not influence their motion, and therefore it
is simply considered as a massless test particle.
The body with the greater mass ``m_1`` is referred as the *primary*, and
``m_2`` as the *secondary*. These bodies are together
called the *primaries* and are assumed
to describe Keplerian circular orbits about their center of mass, which is placed at
the origin of the reference frame. It is further assumed that the orbit of the third body
takes place in the orbital plane of the primaries.
A full treatment of the PCR3BP may be found in [[1]](@ref refsPCR3BP).

The ratio ``\mu = m_2/(m_1+m_2)`` is known as the *mass parameter*. Using
mass units such that ``m_1+m_2=1``, we have ``m_1=1-\mu`` and ``m_2=\mu``. In
this example, we assume the mass parameter to have a value ``\mu=0.01``.
```@example common
using Plots

const μ = 0.01
nothing # hide
```

The Hamiltonian for the PCR3BP in the synodic frame (i.e., a frame which rotates
such that the primaries are at rest on the ``x`` axis) is
```math
H(x, y, p_x, p_y) = \frac{1}{2}(p_x^2+p_y^2) - (x p_y - y p_x) + V(x, y), \tag{1}
```
where
```math
V(x, y) = - \frac{1-\mu}{\sqrt{(x-\mu)^2+y^2}} - \frac{\mu}{\sqrt{(x+1-\mu)^2+y^2}}.\tag{2}
```
is the gravitational potential associated to the primaries. The RHS of Eq.
(1) is also known as the *Jacobi constant*, since it is a
preserved quantity of motion in the PCR3BP. We will use this property to check
the accuracy of the solutions computed.
```@example common
V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
H(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)
H(x) = H(x...)
nothing # hide
```

The equations of motion for the PCR3BP are
```math
\begin{aligned}
    \dot{x} & = p_x + y,\\
    \dot{y} & = p_y - x,\\
    \dot{p_x} & = - \frac{(1-\mu)(x-\mu)}{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu(x+1-\mu)}{((x+1-\mu)^2+y^2)^{3/2}} + p_y,\\
    \dot{p_y} & = - \frac{(1-\mu)y      }{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu y       }{((x+1-\mu)^2+y^2)^{3/2}} - p_x.
\end{aligned}
```

We define this system of ODEs using the most naive approach
```@example common
function f(dq, q, param, t)
    local μ = param[1]
    x, y, px, py = q
    dq[1] = px + y
    dq[2] = py - x
    dq[3] = - (1-μ)*(x-μ)*((x-μ)^2+y^2)^-1.5 - μ*(x+1-μ)*((x+1-μ)^2+y^2)^-1.5 + py
    dq[4] = - (1-μ)*y    *((x-μ)^2+y^2)^-1.5 - μ*y      *((x+1-μ)^2+y^2)^-1.5 - px
    return nothing
end
nothing # hide
```
Note that `DifferentialEquations` offers interesting alternatives to write
these equations of motion in a simpler and more convenient way,
for example, using the macro `@ode_def`,
see [`ParameterizedFunctions.jl`](https://github.com/JuliaDiffEq/ParameterizedFunctions.jl).
We have not used that flexibility here because `TaylorIntegration.jl` has
[`@taylorize`](@ref), which under certain circumstances allows to
important speed-ups.

We shall define the initial conditions ``q_0 = (x_0, y_0, p_{x,0}, p_{y,0})`` such that
``H(q_0) = J_0``, where ``J_0`` is a prescribed value. In order to do this,
we select ``y_0 = p_{x,0} = 0`` and compute the value of ``p_{y,0}`` for which
``H(q_0) = J_0`` holds.

We consider a value for ``J_0`` such that the test particle is
able to display close encounters with *both* primaries, but cannot escape to infinity.
We may obtain a first
approximation to the desired value of ``J_0`` if we plot the projection of the
zero-velocity curves on the ``x``-axis.
```@example common
ZVC(x) =  -x^2/2 + V(x, zero(x)) # projection of the zero-velocity curves on the x-axis

plot(ZVC, -2:0.001:2, label="zero-vel. curve", legend=:topleft)
plot!([-2, 2], [-1.58, -1.58], label="J0 = -1.58")
ylims!(-1.7, -1.45)
xlabel!("x")
ylabel!("J")
title!("Zero-velocity curves (x-axis projection)")
```

Notice that the maxima in the plot correspond to the Lagrangian points ``L_1``, ``L_2``
and ``L_3``; below we shall concentrate in the value ``J_0 = -1.58``.
```@example common
J0 = -1.58
nothing # hide
```

We define a function `py!`, which depends on the initial condition ``q_0 = (x_0, 0, 0, p_{y,0})``
and the Jacobi constant value ``J_0``, such that it computes an adequate value
``p_{y,0}`` for which we have ``H(q_0)=J_0`` and updates (in-place) the initial condition
accordingly.
```@example common
function py!(q0, J0)
    @assert iszero(q0[2]) && iszero(q0[3]) # q0[2] and q0[3] have to be equal to zero
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
We note that the value of `q0` has been updated. We can check that the value of the
Hamiltonian evaluated at the initial condition is indeed equal to `J0`.
```@example common
H(q0) == J0
```

Following the `DifferentialEquations.jl`
[tutorial](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html),
we define an `ODEProblem` for the integration;
`TaylorIntegration.jl` can be used via its common interface bindings with `DiffEqBase.jl`; both packages need to be loaded explicitly.
```@example common
tspan = (0.0, 1000.0)
p = [μ]

using TaylorIntegration, DiffEqBase
prob = ODEProblem(f, q0, tspan, p)
```

We solve `prob` using a 25-th order Taylor method, with a local absolute tolerance ``\epsilon_\mathrm{tol} = 10^{-20}``.
```@example common
solT = solve(prob, TaylorMethod(25), abstol=1e-20);
```

As mentioned above, we load `OrdinaryDiffEq` in order to solve the same problem `prob`
now with the `Vern9` method, which the `DifferentialEquations.jl`
[documentation](http://docs.juliadiffeq.org/latest/solvers/ode_solve.html#Non-Stiff-Problems-1)
recommends for high-accuracy (i.e., very low tolerance) integrations of
non-stiff problems.
```@example common
using OrdinaryDiffEq

solV = solve(prob, Vern9(), abstol=1e-20); #solve `prob` with the `Vern9` method
```

We plot in the ``x-y`` synodic plane the solution obtained
with `TaylorIntegration.jl`:
```@example common
plot(solT, vars=(1, 2), linewidth=1)
scatter!([μ, -1+μ], [0,0], leg=false) # positions of the primaries
xlims!(-1+μ-0.2, 1+μ+0.2)
ylims!(-0.8, 0.8)
xlabel!("x")
ylabel!("y")
```
Note that the orbit obtained displays the expected
dynamics: the test particle explores the regions surrounding both
primaries, located at the red dots, without escaping to infinity.
For comparison, we now plot the orbit corresponding to the
solution obtained with the `Vern9()` integration; note that
the scales are identical.
```@example common
plot(solV, vars=(1, 2), linewidth=1)
scatter!([μ, -1+μ], [0,0], leg=false) # positions of the primaries
xlims!(-1+μ-0.2, 1+μ+0.2)
ylims!(-0.8, 0.8)
xlabel!("x")
ylabel!("y")
```
We note that the orbits do not display the same qualitative features. In particular,
the `Vern9()` integration displays an orbit which does not visit the secondary,
as it was the case in the integration using Taylor's method, and stays far
enough from ``m_1``. The question is which integration should we trust?

We can obtain a quantitative comparison of the validity of both integrations
through the preservation of the Jacobi constant:
```@example common
ET = H.(solT.u)
EV = H.(solV.u)
δET = ET .- J0
δEV = EV .- J0
nothing # hide
```

We plot first the value of the Jacobi constant as function of time.
```@example common
plot(solT.t, H.(solT.u), label="TaylorIntegration.jl")
plot!(solV.t, H.(solV.u), label="Vern9()")
xlabel!("t")
ylabel!("H")
```
Clearly, the integration with `Vern9()` does not conserve the Jacobi constant;
actually, the fact that its value is strongly reduced leads to the artificial trapping
displayed above around ``m_1``. We notice that the loss of conservation of the
Jacobi constant is actually not related to a close approach with ``m_1``.

We now plot, in log scale, the `abs` of the absolute error in the Jacobi
constant as a function of time, for both solutions:
```@example common
plot(solT.t, abs.(δET), yscale=:log10, label="TaylorIntegration.jl")
plot!(solV.t, abs.(δEV), label="Vern9()")
ylims!(10^-18, 10^4)
xlabel!("t")
ylabel!("dE")
```
We notice that the Jacobi constant absolute error for the `TaylorIntegration.jl`
solution remains bounded below ``5\times 10^{-14}``, despite of the fact that
the solution displays many close approaches with ``m_2``.

Finally, we comment on the time spent by each integration.
```@example common
@elapsed solve(prob, TaylorMethod(25), abstol=1e-20);
```
```@example common
@elapsed solve(prob, Vern9(), abstol=1e-20);
```
The integration with `TaylorMethod()` takes *much longer* than that using
`Vern9()`. Yet, as shown above, the former preserves the Jacobi constant
to a high accuracy, whereas the latter
solution loses accuracy in the sense of not conserving the Jacobi constant,
which is an important property to trust the result of the integration.
A fairer comparison is obtained by pushing the native methods of `DiffEqs`
to reach similar accuracy for the integral of motion, as the one
obtained by `TaylorIntegration.jl`. Such comparable situation has
a performance cost, which then makes `TaylorIntegration.jl` comparable
or even faster in some cases; see [[2]](@ref refsPCR3BP).

Finally, as mentioned above, a way to improve the integration time in
`TaylorIntegration` is using the macro [`@taylorize`](@ref); see
[this section](@ref taylorize) for details. Under certain circumstances
it is possible to improve the performance, also with the common interface
with `DifferentialEquations`, which restricts some of the great
flexibility that `DifferentialEquations` allows when writing the function
containing the differential equations.


### [References](@id refsPCR3BP)

[1] Murray, Carl D., Stanley F. Dermott. Solar System dynamics. Cambridge University Press, 1999.

[2] [SciMLBenchmarks.jl/DynamicalODE](https://benchmarks.sciml.ai/html/DynamicalODE/Henon-Heiles_energy_conservation_benchmark.html)
