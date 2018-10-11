# [ODE integration using Taylor's method](@id taylormethod)

Taylor's integration method is a quite powerful method to integrate ODEs
which are smooth enough, allowing to reach a precision comparable
to round-off errors per time-step. A *high-order* Taylor approximation
of the solution (dependent variable) is constructed such that the error
is quite small. A time-step is constructed which guarantees
the validity of the series; this is used to sum up the Taylor
expansion to obtain an approximation of the solution at a later time.


## [The recurrence relation](@id rec_rel)

Let us consider the following
```math
\begin{equation}
\label{eq-ODE}
\dot{x} = f(t, x),
\end{equation}
```
and define the initial value problem with the initial condition
``x(t_0) = x(0)``.

We write the solution of this equation as
```math
\begin{equation}
\label{eq-solution}
x = x_{[0]} + x_{[1]} (t-t_0) + x_{[2]} (t-t_0)^2 + \cdots +
x_{[k]} (t-t_0)^k + \cdots,
\end{equation}
```
where the initial condition imposes that ``x_{[0]} = x(0)``. Below, we show how to
obtain the coefficients ``x_{[k]}`` of the Taylor expansion of the solution.

We assume that the Taylor expansion around ``t_0`` of ``f(t, x(t))`` is known,
which we write as
```math
\begin{equation}
\label{eq-rhs}
f(t, x(t)) = f_{[0]} + f_{[1]} (t-t_0) + f_{[2]} (t-t_0)^2 + \cdots
+ f_{[k]} (t-t_0)^k + \cdots.
\end{equation}
```
Here, ``f_{[0]}=f(t_0,x_0)``, and the Taylor coefficients
``f_{[k]} = f_{[k]}(t_0)`` are the ``k``-th *normalized derivatives* at ``t_0``
given by
```math
\begin{equation}
\label{eq-normderiv}
f_{[k]} = \frac{1}{k!} \frac{{\rm d}^k f} {{\rm d} t^k}(t_0).
\end{equation}
```
Then, we are assuming that we know how to obtain ``f_{[k]}``; these
coefficients are obtained using
[`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl).

Substituting Eq. (\ref{eq-solution}) in (\ref{eq-ODE}), and equating powers
of ``t-t_0``, we obtain
```math
\begin{equation}
\label{eq-recursion}
x_{[k+1]} = \frac{f_{[k]}(t_0)}{k+1}, \quad k=0,1,\dots.
\end{equation}
```
Therefore, the coefficients of the Taylor expansion (\ref{eq-solution})
are obtained *recursively* using Eq. (\ref{eq-recursion}).


## [Time step](@id time-step)

In the computer, the expansion (\ref{eq-solution}) has to be computed
to a finite *order*. We shall denote by ``K`` the order of the series. Clearly,
the larger the order ``K``, the more accurate the obtained solution is.

The theorem of existence and uniqueness of the solution of
Eq.~(\ref{eq-ODE}) ensures that the Taylor expansion converges. Then,
assuming that ``K`` is large enough to be within
the convergent tail. We introduce the parameter ``\epsilon_\textrm{tol} > 0``
to control how large is the last term. The idea is to set this
parameter to a small value, usually smaller than the machine-epsilon.
Denoting by ``h = t_1-t_0`` the time step, then
``| x_{[K]} | h^K \le \epsilon_\textrm{tol}``, we obtain
```math
\begin{equation}
\label{eq-h}
h \le \Big(\frac{\epsilon_\textrm{tol}}{| x_{[K]} |}\Big)^{1/K}.
\end{equation}
```
Equation (\ref{eq-h}) represents the *maximum* time-step which is
consistent with ``\epsilon_\textrm{tol}``, ``K`` and the assumption of
being within the convergence tail. Notice that the arguments exposed
above simply ensure that ``h`` is a maximum time-step, but any other
smaller than ``h`` can be used since the series is convergent in the
open interval ``t\in(t_0-h,t_0+h)``.

Finally, from Eq. (\ref{eq-solution}) with (\ref{eq-h}) we
obtain ``x(t_1) = x(t_0+h)``, which is again an initial value problem.
