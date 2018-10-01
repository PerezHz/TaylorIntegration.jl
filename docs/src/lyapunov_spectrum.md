# [Lyapunov spectrum](@id lyap)

Here we describe the background of Lyapunov spectra computations in
`TaylorIntegration.jl`.

Implementation of Lyapunov spectra computations in `TaylorIntegration.jl`
follow the numerical method of Benettin et al. [1, 2], which itself is based on
Oseledet's multiplicative ergodic theorem [3]. That is, simultaneously to the
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
matrix, which initial condition is selected to be ``\xi(t_0) = I_{n\times n}``,
the ``n\times n`` identity matrix; ``n`` is equal to the degrees of freedom of
the dependent variable ``x``.

For the actual computation of the Lyapunov spectrum, we proceed as follows.
During the simultaneous numerical integration of the equations of motion and the
variational equations, at fixed time intervals ``t_k = k\cdot \Delta t``,
``k = 1, 2, \ldots`` we perform a ``QR`` decomposition over ``\xi(t_k)``, the
solution of the variational equations at time ``t_k``. That is, we factorize
``\xi(t_k)`` as ``\xi(t_k)=Q\cdot R``, where ``Q`` is an orthogonal ``n\times n``
matrix and R is an upper triangular ``n\times n`` matrix with positive diagonal
elements. The diagonal elements ``R_{ii,k}`` associated to the matrix ``R``,
obtained from the ``QR`` process at time ``t_k``, allow us to obtain the growth
factors from which the ``l``-th Lyapunov exponent is computed at time ``t_k`` as
```math
\begin{equation}
\label{lyap-spec}
\lambda_l = \sum_{m=1}^k \frac{\log (R_{ll,m})}{k\cdot \Delta t}.
\end{equation}
```
On the other hand, the components of the ``Q`` matrix are substituted back into
``\xi(t_k)``. The equations of motion, together with the variational equations,
are then integrated up to time ``t_{k+1}``, and the process is repeated for
sufficiently large ``k``, until convergence up to a prescribed tolerance is
reached.

## References

[1] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980a, Meccanica, 15, 9

[2] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980b, Meccanica, 15, 21

[3] Oseledets V. I., 1968, Trudy Moskovskogo Matematicheskogo Obshchestva, 19, 179