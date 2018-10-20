# [Lyapunov spectrum](@id lyap)

Here we describe the background of the Lyapunov spectra computations in
`TaylorIntegration.jl`. Our implementation follows the numerical method
of Benettin et al. [[1], [2]](@ref refsL),
which itself is based on
Oseledet's multiplicative ergodic theorem [[3]](@ref refsL). Namely, simultaneously to the
integration of the equations of motion, we integrate the 1st-order variational
equations associated to them.

In general, given a dynamical system defined by the equations of motion
```math
\begin{equation}
\label{eq-ODE-l}
\dot{x} = f(t, x),
\end{equation}
```
along with the initial condition ``x(t_0) = x_0``, then the first-order
variational equations associated to this system are
```math
\begin{equation}
\label{var-eqs}
\dot{\xi} = (\operatorname{D}f)(x(t))\cdot \xi,
\end{equation}
```
where ``(\operatorname{D}f)(x(t))`` is the Jacobian of the function ``f`` with
respect to the dependent variable ``x``, evaluated at time ``t``, for a given
solution ``x(t)`` to the equations of motion. The variable ``\xi`` denotes a
matrix, whose initial condition is ``\xi(t_0) = \mathbb{1}_{n}``,
the ``n\times n`` identity matrix, where ``n`` is the degrees of freedom or number
of dependent variables ``x``.

For the actual computation of the Lyapunov spectrum, we proceed as follows.
During the simultaneous numerical integration of the equations of motion and the
variational equations, at fixed time intervals ``t_k = k\cdot \Delta t``,
``k = 1, 2, \ldots`` we perform a ``QR`` decomposition over ``\xi(t_k)``, the
solution of the variational equations at time ``t_k``. That is, we factorize
``\xi(t_k)`` as ``\xi(t_k)=Q_k\cdot R_k``, where ``Q_k`` is an orthogonal ``n\times n``
matrix and ``R_k`` is an upper triangular ``n\times n`` matrix with positive diagonal
elements. The diagonal elements ``R_{ii,k}`` are the growth
factors from which the ``l``-th Lyapunov exponent is computed at time ``t_k``
```math
\begin{equation}
\label{lyap-spec}
\lambda_l = \sum_{m=1}^k \frac{\log (R_{ll,m})}{k\cdot \Delta t}.
\end{equation}
```
In turn, the matrix ``Q`` is substituted into ``\xi(t_k)`` as the new (scaled)
initial condition.

The equations of motion together with the variational equations are integrated
up to time ``t_{k+1}`` using Taylor's method. We note that each [time step](@ref time-step)
of the integration is determined using the normalized derivatives of ``x`` and the
tolerance ``\epsilon_\textrm{tol}``. This process is repeated until a prescribed
``t_\textrm{max}`` is reached.

[This example](@ref lyap_lorenz) illustrates the computation
of the Lyapunov spectrum for the Lorenz system.


### [References](@id refsL)

[1] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980, Meccanica, 15, 9

[2] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980, Meccanica, 15, 21

[3] Oseledets V. I., 1968, Trudy Moskovskogo Matematicheskogo Obshchestva, 19, 179
