# [Interoperability with `DifferentialEquations.jl`](@id diffeqinterface)

Here, we show an example of interoperability between `TaylorIntegration.jl` and
[`DifferentialEquations.jl`](https://github.com/JuliaDiffEq/DifferentialEquations.jl), i.e.,
how to use `TaylorIntegration.jl` from the `DifferentialEquations` ecosystem. The basic
requirement is to load `OrdinaryDiffEq.jl` together with `TaylorIntegration.jl`, which
sets-up the common interface. Below, we shall also use `OrdinaryDiffEq.jl` to compare
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

We define this system of ODEs in a way that allows the use of the [`@taylorize`](@ref)
macro from `TaylorIntegration.jl`, which for the present example allows
important speed-ups. For more details about the specifics of the use of [`@taylorize`](@ref),
see [this section](@ref taylorize).
```@example common
using TaylorIntegration
@taylorize function pcr3bp!(dq, q, param, t)
    local μ = param[1]
    local onemμ = 1 - μ
    x1 = q[1]-μ
    x1sq = x1^2
    y = q[2]
    ysq = y^2
    r1_1p5 = (x1sq+ysq)^1.5
    x2 = q[1]+onemμ
    x2sq = x2^2
    r2_1p5 = (x2sq+ysq)^1.5
    dq[1] = q[3] + q[2]
    dq[2] = q[4] - q[1]
    dq[3] = (-((onemμ*x1)/r1_1p5) - ((μ*x2)/r2_1p5)) + q[4]
    dq[4] = (-((onemμ*y )/r1_1p5) - ((μ*y )/r2_1p5)) - q[3]
    return nothing
end
nothing # hide
```

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

plot(ZVC, -2:0.001:2, label="zero-vel. curve", legend=:topleft, fmt = :png)
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
[tutorial](https://diffeq.sciml.ai/stable/tutorials/ode_example/),
we define an `ODEProblem` for the integration; `TaylorIntegration` can be used
via its common interface bindings with `OrdinaryDiffEq.jl`; both packages need to
be loaded explicitly.
```@example common
tspan = (0.0, 2000.0)
p = [μ]

using OrdinaryDiffEq
prob = ODEProblem(pcr3bp!, q0, tspan, p)
```

We solve `prob` using a 25-th order Taylor method, with a local absolute tolerance ``\epsilon_\mathrm{tol} = 10^{-15}``.
```@example common
solT = solve(prob, TaylorMethod(25), abstol=1e-15);
```

As mentioned above, we will now solve the same problem `prob`
with the `Vern9` method from `OrdinaryDiffEq`, which the `DifferentialEquations`
[documentation](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Non-Stiff-Problems)
recommends for high-accuracy (i.e., very low tolerance) integrations of
non-stiff problems. Note that, besides setting an absolute tolerance `abstol=1e-15`,
we're setting a relative tolerance `reltol=1e-15` [[2]](@ref refsPCR3BP). We have found that for the
current problem this is a good balance between speed and accuracy for the `Vern9`
method, i.e., the `Vern9` integration becomes noticeably slower (although more
accurate) if either `abstol` or `reltol` are set to lower values.
```@example common
using OrdinaryDiffEq

solV = solve(prob, Vern9(), abstol=1e-15, reltol=1e-15); #solve `prob` with the `Vern9` method
```

We plot in the ``x-y`` synodic plane the solution obtained
with `TaylorIntegration`:
```@example common
plot(solT, vars=(1, 2), linewidth=1, fmt = :png)
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
plot(solV, vars=(1, 2), linewidth=1, fmt = :png)
scatter!([μ, -1+μ], [0,0], leg=false) # positions of the primaries
xlims!(-1+μ-0.2, 1+μ+0.2)
ylims!(-0.8, 0.8)
xlabel!("x")
ylabel!("y")
```
We note that both orbits display the same qualitative features, and also some
differences. For instance, the `TaylorMethod(25)` solution gets closer to the
primary than that the `Vern9()`. We can obtain a
quantitative comparison of the validity of both integrations
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
plot(solT.t, H.(solT.u), label="TaylorMethod(25)", fmt = :png, yformatter = :plain)
plot!(solV.t, H.(solV.u), label="Vern9()")
xlabel!("t")
ylabel!("H")
```
In the scale shown we observe that, while both solutions
display a preservation of the Jacobi constant to a certain degree, the `Vern9()`
solution suffers sudden jumps during the integration.

We now plot, in log scale, the `abs` of the absolute error in the Jacobi
constant as a function of time, for both solutions:
```@example common
plot(solT.t, abs.(δET), yscale=:log10, label="TaylorMethod(25)", legend=:topleft, fmt = :png, yformatter = :plain)
plot!(solV.t, abs.(δEV), label="Vern9()")
ylims!(10^-16, 10^-10)
xlabel!("t")
ylabel!("dE")
```
We notice that the Jacobi constant absolute error for the `TaylorMethod(25)`
solution remains bounded below ``10^{-13}`` throughout the integration. While the `Vern9()`
solution at the end of the integration time has reached a similar value, it displays a
larger Jacobi constant error earlier in time.

Finally, we comment on the time spent by each integration.
```@example common
using BenchmarkTools
bT = @benchmark solve($prob, $(TaylorMethod(25)), abstol=1e-15)
bV = @benchmark solve($prob, $(Vern9()), abstol=1e-15, reltol=1e-15)
nothing # hide
```
```@example common
bT # TaylorMethod(25) benchmark
```
```@example common
bV # Vern9 benchmark
```
We notice in this setup, where the `TaylorMethod(25)` and the `Vern9()` integrations perform
similarly in terms of accuracy, the former performs better in terms of runtime.

We can tune the `abstol` and `reltol` for the `Vern9()` method we
so that performance is similar. Such
situation has an accuracy cost, which then makes `TaylorIntegration`
a sensible alternative for high-accuracy integrations of non-stiff ODEs in some cases; see
[[2]](@ref refsPCR3BP).

Finally, as mentioned above, a crucial way in which `TaylorIntegration`
provides high accuracy at competitive speeds is through the use of
the [`@taylorize`](@ref) macro; see
[this section](@ref taylorize) for details. Currently, `TaylorIntegration`
supports the use of `@taylorize` via the common interface with
`DifferentialEquations` only for in-place `ODEProblem`.


### [References and notes](@id refsPCR3BP)

[1] Murray, Carl D., Stanley F. Dermott. Solar System dynamics. Cambridge University Press, 1999.

[2] [SciMLBenchmarks.jl/DynamicalODE](https://benchmarks.sciml.ai/html/DynamicalODE/Henon-Heiles_energy_conservation_benchmark.html)
