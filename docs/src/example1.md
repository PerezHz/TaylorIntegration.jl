# Examples

Here, we shall illustrate first with a simple example how the method
explicitly constructs the solution, and then how to use the package
itself.

## Illustration of the method

To illustrate the method, we shall consider the differential equation
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
\begin{eqnarray*}
x(t) & = & x_0 + x_0^2 t + x_0^3 t^2 + \cdots + x_0^{k+1} t^k + \cdots \\
& = & \frac{x_0}{1-x_0 t}.
\end{eqnarray*}
```

We note that the solution computed is identical to the exact solution,
Eq. (\ref{eq-sol1}), recognizing that the series is the geometric series.

We note that the solution written as a series does not manifest in an
that the solution is only defined for ``t<1/x_0``. Yet, considering the
actual formulation of the Picard-Lindelöf theorem for ordinary
differential equations (existence and uniqueness of the solution), we
can actually learn about this. The theorem imposes that the RHS of the
differential equation must satisfy a Lipschitz condition. In this case,
considering ``A>0``, the RHS ``f(x) = x^2`` over the interval
``I_A=(-A,A)`` satisfies the Lipschitz condition
``|f(x)-f(y)|=|x^2-y^2|=|x+y|\cdot|x-y| \leq 2A|x-y|``
for every ``x, y \in I_A``. Therefore, the theorem guarantees the
existence and uniqueness of the solution for $t \in [0,\delta]$, for any
$x_0 \in I_A$. Notice that in this case, it is necessary to restrict the
function to a bounded interval in order to fulfill a Lipschitz condition.


# Infinity in finite time

```@meta
CurrentModule = TaylorIntegration
```
We shall illustrate how to use `TaylorIntegration.jl` to integrate
Eq. (\ref{eq-example1}) for the initial condition ``x(0)=3``. Notice
that according to the exact solution Eq. (\ref{eq-sol1}), the solution
only exists for ``t<t_\mathrm{max} =1/3``; in addition, we note that
this number can not be represented exactly as a floating-point.

We first load the required packages and define a function which
represents the equation of motion.

```@repl example1
using TaylorIntegration, Plots, LaTeXStrings
diffeq(t, x) = x.^2
```

In `TaylorIntegration.jl` we follow the convention to write the
argument of the function with the equations of motion using first
the independent variable (``t``), then the dependent variables (``x``)
and then the derivatives defining the equations of motion.

Now, we integrate the equations of motion using [`taylorinteg`](@ref); despite
of the fact that the solution only exists for ``t<t\mathrm{max}``, below
we *try* to compute up to ``t_\mathrm{end}=0.34``. For the integration
presented below, we use a 28-th series expansion, with
``\epsilon_\textrm{tol} = 10^{-20}``

```@repl example1
tT, xT = taylorinteg(diffeq, 3.0, 0.0, 0.34, 28, 1e-20, maxsteps=200);
```

With this parameters, we first note that the end points of the
calculation did not exceed ``t_\textrm{max}``
```@repl example1
tT[end], xT[end]
```

~~~~
Error: UndefVarError: tT not defined
~~~~




How many steps did the Taylor integrator perform?

~~~~{.julia}
length(xT)-1
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: xT not defined
~~~~




Below, we show the $x$ vs $t$ plot (log-log):

~~~~{.julia}
plot(log10(tT[2:end]), log10(xT[2:end]))
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: plot not defined
~~~~



~~~~{.julia}
title!("x vs t (log-log)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: title! not defined
~~~~



~~~~{.julia}
xlabel!(L"\log_{10}(t)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylabel!(L"\log_{10}(x(t))")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~




Does the integrator get past the singularity?

~~~~{.julia}
tT[end] > 1/3
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: tT not defined
~~~~




The answer is no! Even if we increase the value of the `maxsteps` keyword in `taylorinteg`, it doesn't get past the singularity!


Now, the relative difference between the numerical and analytical solution, $\delta x$, is:

~~~~{.julia}
δxT = (xT.-exactsol(tT, 3.0))./exactsol(tT, 3.0);
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: tT not defined
~~~~




The $\delta x$ vs $t$ plot (logscale):

~~~~{.julia}
plot(tT[6:end], log10(abs(δxT[6:end])))
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: plot not defined
~~~~



~~~~{.julia}
title!("Relative error (semi-log)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: title! not defined
~~~~



~~~~{.julia}
xlabel!(L"t")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylabel!(L"\log_{10}(\delta x(t))")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~




We observe that, while the execution time is ~10 times longer wrt 4th-order RK, the numerical solution obtained by the Taylor integrator stays within $10^{-12}$ of the analytical solution, for a same number of steps.

Now, that happens if we use a higher order Runge Kutta method to integrate this problem?


## 3. Runge-Kutta-Fehlberg 7/8 method


Here we use the Runge-Kutta-Fehlberg 7/8 method, included in `ODE.jl`, to integrate the same problem as before.

~~~~{.julia}
@time t78, x78 = ode78(diffeq, 3.0, [0.0, 0.34]); #warmup lap
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: ode78 not defined
~~~~



~~~~{.julia}
@time t78, x78 = ode78(diffeq, 3.0, [0.0, 0.34]);
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: ode78 not defined
~~~~




Plot $x$ vs $t$ (log-log):

~~~~{.julia}
plot(log10(t78[2:end]), log10(x78[2:end]))
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: plot not defined
~~~~



~~~~{.julia}
title!("x vs t (log-log)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: title! not defined
~~~~



~~~~{.julia}
xlabel!(L"\log_{10}(t)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylabel!(L"\log_{10}(x(t))")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~




What is the final state of the system?

~~~~{.julia}
t78[end], x78[end]
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: t78 not defined
~~~~




Does the integrator get past the singularity?

~~~~{.julia}
t78[end]>1/3
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: t78 not defined
~~~~




The answer is yes! So the last value of the solution is meaningless:

~~~~{.julia}
x78[end] #this value is meaningless
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: x78 not defined
~~~~




How many steps did the RK integrator perform?

~~~~{.julia}
length(x78)-1
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: x78 not defined
~~~~




The relative difference between the numerical and analytical solution, $\delta x$, is:

~~~~{.julia}
δx78 = (x78-exactsol(t78, 3.0))./exactsol(t78, 3.0) #error relative to analytical solution
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: x78 not defined
~~~~



~~~~{.julia}
;
~~~~~~~~~~~~~




The $\delta x$ vs $t$ plot (semilog):

~~~~{.julia}
plot(t78[2:end], log10(abs(δx78[2:end])))
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: plot not defined
~~~~



~~~~{.julia}
title!("Relative error (semi-log)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: title! not defined
~~~~



~~~~{.julia}
xlabel!(L"t")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylabel!(L"\log_{10}(\delta x(t))")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~




This time, the RKF 7/8 integrator is "only" twice as fast as the Taylor integrator, but the error continues to be greater than the error from the latter by several orders of magnitude.


## 4. Adaptive 4th-order RK, stringer tolerance


As a last example, we will integrate once again our problem using a 4th-order adaptive RK integrator, but imposing a stringer tolerance:

~~~~{.julia}
@time tRK_, xRK_ = ode45(diffeq, 3.0, [0.0, 0.34], abstol=1e-8, reltol=1e-8 ); #warmup lap
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: ode45 not defined
~~~~



~~~~{.julia}
@time tRK_, xRK_ = ode45(diffeq, 3.0, [0.0, 0.34], abstol=1e-8, reltol=1e-8 );
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: ode45 not defined
~~~~



~~~~{.julia}
;
~~~~~~~~~~~~~




Now, the integrator takes 10 times longer to complete the integration than the Taylor method.


Does it get past the singularity?

~~~~{.julia}
tRK_[end] > 1/3
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: tRK_ not defined
~~~~




Yes! So, once again, the last value reported by the integrator is completely meaningless. But, has it attained a higher precision than the Taylor method? Well, let's calculate once again the numerical error relative to the analytical solution:

~~~~{.julia}
δxRK_ = (xRK_-exactsol(tRK_, 3.0))./exactsol(tRK_, 3.0);
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: xRK_ not defined
~~~~




And now, let's plot this relative error vs time:

~~~~{.julia}
plot(tRK_[2:end], log10(abs(δxRK_[2:end])))
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: plot not defined
~~~~



~~~~{.julia}
title!("Relative error (semi-log)")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: title! not defined
~~~~



~~~~{.julia}
xlabel!(L"t")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylabel!(L"\log_{10}(\delta x(t))")
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: @L_str not defined
~~~~



~~~~{.julia}
ylims!(-20,20)
~~~~~~~~~~~~~


~~~~
Error: UndefVarError: ylims! not defined
~~~~




The numerical error has actually gotten worse! `TaylorIntegration.jl` is indeed a really competitive package to integrate ODEs.
