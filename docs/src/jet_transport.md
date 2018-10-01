# [Jet transport](@id jettransport)

In this section we describe jet transport in `TaylorIntegration.jl`.

*Jet transport* is a tool that allows the propagation under the flow of a small
neighborhood in phase space around a given initial condition, instead of
propagating single initial conditions only.

To compute the propagation of ``\mathbf{x}_0 + \delta \mathbf{x}``, where
``\delta \mathbf{x}`` are independent small displacements in phase space, one has
to solve high-order variational equations. The idea is to treat
``\mathbf{x}_0 + \delta \mathbf{x}`` as a (truncated) polynomial in the
``\delta \mathbf{x}`` variables.

Jet transport works in general with *any* ordinary ODE solver, provided the
chosen solver supports computations using *multi-variate polynomial algebra*.

## An example

From D. Pérez-Palau et al, *Celest. Mech, Dyn. Astron.* **123**, 239-262 (2015).

Let us consider the differential equations for the harmonic oscillator:
```math
\begin{eqnarray*}
\dot{x} & = & y, \\
\dot{y} & = & -x,
\end{eqnarray*}
```
with the initial condition ``\mathbf{x}_0=[x_0, y_0]^T``.

Euler's method corresponds to:
```math
\begin{equation*}
\mathbf{x}_{n+1} = \mathbf{x}_n + h \mathbf{f}(\mathbf{x}_n).
\end{equation*}
```
Instead of the initial condition, we consider the polynomial
```math
\begin{equation*}
P_{0,\mathbf{x}_0}(\delta\mathbf{x}) = [x_0+\delta x, y_0 + \delta y]^T.
\end{equation*}.
```
Then,
```math
\begin{eqnarray*}
\mathbf{x}_1 &=& P_{h, \mathbf{x}_0}(\delta\mathbf{x}) =
P_{0,\mathbf{x}_0}(\delta\mathbf{x}) + h\, \mathbf{f}(P_{0,\mathbf{x}_0}(\delta\mathbf{x}))\\ & = &
\left(
\begin{array}{c}
x_0 + h y_0 \\ y_0 - h x_0
\end{array}
\right) + \left(
\begin{array}{cc}
1 & h \\ -h & 1
\end{array}
\right) \left(
\begin{array}{c}
\delta x\\ \delta y
\end{array}
\right).
\end{eqnarray*}
```

At each step of Taylor's method, we can write
```math
\begin{equation*}
\phi(t_n+h; \mathbf{x}_n) = \sum_{i=0}^p \mathbf{x}^{(i)}(\mathbf{x}_n, t_n) \,h^i.
\end{equation*}
```
