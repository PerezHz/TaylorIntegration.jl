# [Jet transport](@id jettransport)

In this section we describe jet transport in `TaylorIntegration.jl`.

*Jet transport* is a tool that allows the propagation under the flow of a small
neighborhood in phase space around a given initial condition, instead of
propagating single initial conditions only.

To compute the propagation of ``\vec{x}_0 + \delta \vec{x}``, where
``\delta \vec{x}``are independent small displacements in phase space, one has
to solve high-order variational equations. The idea is to treat
``\vec{x}_0 + \delta \vec{x}`` as a (truncated) polynomial in the
``\delta \vec{x}`` variables.

Jet transport works in general with *any* ordinary ODE solver, provided the
chosen solver supports computations using *multi-variate polynomial algebra*.

## An example,

From D. Pérez-Palau et al, *Celest. Mech, Dyn. Astron.* **123**, 239-262 (2015).

Let us consider the differential equations for the harmonic oscillator:
```math
\begin{eqnarray*}
\dot{x} & = & y, \\
\dot{y} & = & -x,
\end{eqnarray*}
```
with the initial condition ``\mathbf{x}_0=[x_0, y_0]^T``.



