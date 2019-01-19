# [Infinity in finite time](@id example1)


## Illustration of the method

We shall illustrate first with a simple example how the method
explicitly constructs the solution, and how to use the package
to obtain it.

We consider the differential equation given by
```math
\begin{equation}
\label{eq-example1}
\dot{x} = x^2,
\end{equation}
```
with the initial condition ``x(0)=x_0``, whose exact solution reads
```math
\begin{equation}
\label{eq-sol1}
x(t) = \frac{x_0}{1-x_0t}.
\end{equation}
```
We shall implement the construction of this example explicitly, which
illustrates the way `TaylorIntegration.jl` is conceived.

The initial condition defines the 0-th order approximation, i.e.,
``x(t) = x_0 + \mathcal{O}(t^1)``. We now write the solution as
``x(t) = x_0 + x_{[1]}t + \mathcal{O}(t^2)``, and we want to determine
``x_{[1]}``. Substituting this solution into the RHS of (\ref{eq-example1}),
yields
```math
x^2 = x_0^2 + 2 x_0 x_{[1]} t + x_{[1]}^2 t^2 =
 x_0^2 + \mathcal{O}(t^1),
```
where in the last equality we have kept all terms up to order 0, since we want
to determine ``x_{[1]}``, and the recursion formula requires for that the 0-th
order term of the Taylor expansion of the RHS of the equation of motion.
Hence, we have ``f_{[0]}=x_0^2``, which using the recursion relation
``x_{[k+1]} = f_{[k]}/(k+1)`` yields ``x_{[1]} = x_0^2``.

Following the same procedure, we write
``x(t) = x_0 + x_{[1]} t + x_{[2]} t^2 + \mathcal{O}(t^3)``, and
```math
x^2 = x_0^2 + 2 x_0 x_{[1]} t + \mathcal{O}(t^2),
```
where we kept all terms up to order 1. We thus have
``f_{[1]}=2 x_0 x_{[1]} = 2 x_0^3``, which then yields ``x_{[2]} = x_0^3``.
Repeating this calculation, we obtain
```math
\begin{equation}
\label{eq-solTaylor}
x(t) = x_0 + x_0^2 t + x_0^3 t^2 + \cdots + x_0^{k+1} t^k + \cdots.
\end{equation}
```

The solution given by Eq. (\ref{eq-solTaylor}) is a geometrical
series, which is identical to the exact solution, Eq. (\ref{eq-sol1}).
Yet, it is not obvious from the solution that it is only defined
for ``t<1/x_0``. To see this, we obtain the step size, as described
previously, for the series truncated to order ``k``.
The Taylor coefficient of order ``k`` is ``x_{[k]}=x_0^{k+1}``,
so the time step is
```math
h < \Big(\frac{\epsilon_\textrm{tol}}{|x_0^{k+1}|}\Big)^{1/k} =
\frac{\epsilon_\textrm{tol}^{1/k}}{|x_0|^{1+1/k}}.
```

In the limit ``k\to\infty`` we obtain ``h < h_\textrm{max}=1/x_0``,
which is the domain of existence of the exact solution.

Below, we shall fix a maximum order for the expansion. This entails
a truncation error which is somewhat controlled through the
absolute tolerance ``\epsilon_\textrm{tol}``. The key to a correct
use of Taylor's method is to impose a quite small value of
``\epsilon_\textrm{tol}`` *together* with a large enough order
of the expansion.


## [Implementation](@id implementation_ex1)

We shall illustrate how to use `TaylorIntegration.jl` to integrate
Eq. (\ref{eq-example1}) for the initial condition ``x(0)=3``. Notice
that according to the exact solution Eq. (\ref{eq-sol1}), the solution
only exists for ``t<t_\mathrm{max} =1/3``; in addition, we note that
this number can not be represented exactly as a floating-point.

We first load the required packages and define a function which
represents the equation of motion.

```@example example1
using TaylorIntegration, Plots
diffeq(t, x) = x^2;
```

!!! note
    In `TaylorIntegration.jl`, the convention for writing the
    function representing the equations of motion is to use first the
    independent variable (`t`), followed by the dependent variables (`x`)
    and then the derivatives defining the equations of motion (`dx`).
    For a single ODE, as in the present case, we omit the last argument
    which is returned, and avoid using vectors; for more ODEs, both `x` and `dx`
    are preallocated vectors and the function mutates (modifies) `dx`.

Now, we integrate the equations of motion using [`taylorinteg`](@ref);
despite of the fact that the solution only exists for ``t<t_\textrm{max}``,
below we shall *try* to compute it up to ``t_\textrm{end}=0.34``; as we shall
see, Taylor's method takes care of this. For
the integration presented below, we use a 28-th series expansion, with
``\epsilon_\textrm{tol} = 10^{-20}``, and compute up to 150
integration steps.

```@example example1
tT, xT = taylorinteg(diffeq, 3.0, 0.0, 0.34, 28, 1e-20, maxsteps=150);
```

We first note that the last point of the
calculation does not exceed ``t_\textrm{max}``.
```@example example1
tT[end]
```
Increasing the `maxsteps` parameter pushes `tT[end]` closer to ``t_\textrm{max}``
but it actually does not reach this value.

Figure 1 displays the computed solution as a function of
time, in log scale.
```@example example1
plot(tT, log10.(xT), shape=:circle)
xlabel!("t")
ylabel!("log10(x(t))")
xlims!(0,0.34)
title!("Fig. 1")
```

Clearly, the solution diverges without bound when
``t\to t_\textrm{max} = 1/3``, i.e., ``x(t)`` approaches infinity in
finite time.

Figure 2 shows the relative difference between the numerical
and the analytical solution in terms of time.

```@example example1
exactsol(t, x0) = x0 / (1 - x0 * t)
δxT = abs.(xT .- exactsol.(tT, 3.0)) ./ exactsol.(tT, 3.0);
plot(tT[6:end], log10.(δxT[6:end]), shape=:circle)
xlabel!("t")
ylabel!("log10(dx(t))")
xlims!(0, 0.4)
title!("Fig. 2")
```

To put in perspective how good is the constructed solution, we
impose (arbitrarily) a relative accuracy of ``10^{-13}``; the time until
such accuracy is satisfied is given by:
```@example example1
indx = findfirst(δxT .> 1.0e-13);
esol = exactsol(tT[indx-1],3.0);
tT[indx-1], esol, eps(esol)
```
Note that, the accuracy imposed in terms of the actual value
of the exact solution means that the difference of the computed
and the exact solution is essentially due to the `eps` of the
computed value.
