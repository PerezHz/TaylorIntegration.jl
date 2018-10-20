# [Jet transport](@id jettransport)

In this section we describe the jet transport capabilities included in `TaylorIntegration.jl`.
*Jet transport* is a tool that allows the propagation under the flow of a small
neighborhood in phase space around a given initial condition, instead of
propagating a single initial condition only.

To compute the propagation of ``\mathbf{x}_0 + \delta \mathbf{x}``, where
``\delta \mathbf{x}`` are independent small displacements in phase space around
the initial condition ``\mathbf{x}_0``, one has
to solve high-order variational equations. The idea is to treat
``\mathbf{x}_0 + \delta \mathbf{x}`` as a truncated polynomial in the
``\delta \mathbf{x}`` variables. The maximum order of this polynomial
has to be fixed in advance.

Jet transport works in general with *any* ordinary ODE solver, provided the
chosen solver supports computations using *multi-variate polynomial algebra*.


## A simple example

Following D. Pérez-Palau et al [[1]](@ref refsJT),
let us consider the differential equations for the harmonic oscillator:
```math
\begin{eqnarray*}
\dot{x} & = & y, \\
\dot{y} & = & -x,
\end{eqnarray*}
```
with the initial condition ``\mathbf{x}_0=[x_0, y_0]^T``.
We illustrate jet transport techniques using Euler's method
```math
\begin{equation*}
\mathbf{x}_{n+1} = \mathbf{x}_n + h \mathbf{f}(\mathbf{x}_n).
\end{equation*}
```

Instead of considering the initial conditions ``\mathbf{x}_0``, we consider
the time evolution of the polynomial
```math
\begin{equation*}
P_{0,\mathbf{x}_0}(\delta\mathbf{x}) = [x_0+\delta x, y_0 + \delta y]^T,
\end{equation*}
```
where ``\delta x`` and ``\delta y`` are small displacements. Below we
concentrate in polynomials of order 1 in ``\delta x`` and ``\delta y``; since
the equations of motion of the harmonic oscillator are linear, there are
no higher order terms.

Using Euler's method we obtain
```math
\begin{eqnarray*}
  \mathbf{x}_1 & = &
  \left(
    \begin{array}{c}
    x_0 + h y_0 \\
    y_0 - h x_0
    \end{array}
  \right)
  + \left(
      \begin{array}{cc}
         1 & h \\
        -h & 1
      \end{array}
    \right)
    \left(
      \begin{array}{c}
        \delta x\\
        \delta y
      \end{array}
    \right). \\
  \mathbf{x}_2 & = &
  \left(
    \begin{array}{c}
    1-h^2 x_0 + 2 h y_0 \\
    1-h^2 y_0 - 2 h x_0
    \end{array}
  \right)
  + \left(
    \begin{array}{cc}
      1-h^2 & 2 h \\
      -2 h & 1-h^2
    \end{array}
    \right)
    \left(
      \begin{array}{c}
        \delta x\\
        \delta y
      \end{array}
    \right).
\end{eqnarray*}
```

The first terms in the expressions for ``\mathbf{x}_1`` and ``\mathbf{x}_2``
above
correspond to the result of an Euler integration step using the initial conditions
only. The other terms are the (linear) corrections which involve the small
displacements ``\delta x`` and ``\delta y``.

In general, for differential equations involving non-linear terms, the resulting
expansions in ``\delta x`` and ``\delta y`` will reflect aspects of the
non-linearities of the ODEs. Clearly, jet transport techniques allow to address
stability properties beyond the linear case, though memory constraints may
play a role. See [this example](@ref pendulum) illustrating the
implementation for the simple pendulum, and [this one](@ref jettransport2)
illustrating the construction of a Poincaré map with Jet transport techniques.


### [References](@id refsJT)

[1] D. Pérez-Palau, Josep J. Masdemont, Gerard Gómez, 2015, Celest. Mech. Dyn. Astron.
123, 239.
