# [Lyapunov spectrum of Lorenz system](@id lyap_lorenz)

Here, we present the calculation of the Lyapunov spectrum of the
[Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system),
using `TaylorIntegration.jl`. The computation involves evaluating the 1st
order variational equations ``\dot \xi = J \cdot \xi`` for this system, where
``J = \operatorname{D}f`` is the Jacobian. By default, the numerical value of
the Jacobian is computed using automatic differentiation techniques
implemented in [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl),
which saves us from writing down explicitly the Jacobian. Conversely, this can be
used to check a function implementing the Jacobian. As an alternative,
specially important if performance is critical, the user may provide a Jacobian
function.

The Lorenz system is the ODE defined as:
```math
\begin{eqnarray*}
    \dot{x}_1 & = & \sigma(x_2-x_1), \\
    \dot{x}_2 & = & x_1(\rho-x_3)-x_2, \\
    \dot{x}_3 & = & x_1x_2-\beta x_3,
\end{eqnarray*}
```
where ``\sigma``, ``\rho`` and ``\beta`` are constant parameters.

First, we write a Julia function which evaluates (in-place) the Lorenz system:
```@example lorenz
#Lorenz system ODE:
function lorenz!(dq, q, params, t)
    σ, ρ, β = params
    x, y, z = q
    dq[1] = σ*(y-x)
    dq[2] = x*(ρ-z)-y
    dq[3] = x*y-β*z
    nothing
end
nothing #hide
```
Below, we use the the parameters ``\sigma = 16.0``, ``\beta = 4`` and
``\rho = 45.92``.
```@example lorenz
#Lorenz system parameters
#we use the `const` prefix in order to help the compiler speed things up
const params = [16.0, 45.92, 4.0]
nothing # hide
```

We define the initial conditions, the initial and final integration times
for the integration:
```@example lorenz
const x0 = [19.0, 20.0, 50.0] #the initial condition
const t0 = 0.0     #the initial time
const tmax = 100.0 #final time of integration
nothing # hide
```

Since the diagonal of the Jacobian is constant, the sum of the Lyapunov spectrum
has to be equal to that value. We calculate this trace using
`TaylorSeries.jl`, and after the numerical integration, we will come back to
check if this value is conserved (or approximately conserved)
as a function of time.
```@example lorenz
# Note that TaylorSeries.jl is @reexport-ed by TaylorIntegration.jl
# Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
import LinearAlgebra: tr
using TaylorIntegration
xi = set_variables("δ", order=1, numvars=length(x0))
x0TN = x0 .+ xi
dx0TN = similar(x0TN)
lorenz!(dx0TN, x0TN, params, t0)
jjac = TaylorSeries.jacobian(dx0TN)
lorenztr = tr(jjac) #trace of Lorenz system Jacobian matrix
nothing # hide
```

As explained above, the user may provide a function which computes the Jacobian
of the ODE in-place:
```@example lorenz
#Lorenz system Jacobian (in-place):
function lorenz_jac!(jac, x, params, t)
    σ, ρ, β = params
    jac[1,1] = -σ + zero(x[1])
    jac[1,2] = σ + zero(x[1])
    jac[1,3] = zero(x[1])
    jac[2,1] = ρ - x[3]
    jac[2,2] = -1.0 + zero(x[1])
    jac[2,3] = -x[1]
    jac[3,1] = x[2]
    jac[3,2] = x[1]
    jac[3,3] = -β + zero(x[1])
    nothing
end
nothing # hide
```
!!! note
    We use of `zero(x[1])` in the function `lorenz_jac!` when the RHS consists
    of a numeric value; this is needed to allow the proper promotion of the
    variables to carry out Taylor's method.

We can actually check the consistency of `lorenz_jac!` with the computation
of the jacobian using automatic differentiation techniques. Below we use
the initial conditions `x0`, but it is easy to generalize this.
```@example lorenz
lorenz_jac!(jjac, x0, params, t0)  # update the matrix `jjac` using Jacobian provided by the user
TaylorSeries.jacobian(dx0TN) == jjac    # `dx0TN` is obtained via automatic differentiation
```

Now, we are ready to perform the integration using `lyap_taylorinteg` function,
which integrates the
1st variational equations and uses Oseledets' theorem. The expansion order will
be ``28`` and the local absolute tolerance will be ``10^{-20}``.
`lyap_taylorinteg` will return three arrays: one with the evaluation times, one
with the values of the dependent variables (at the time of evaluation), and another
one with the values of the Lyapunov spectrum.

We first carry out the integration computing internally the Jacobian
```@example lorenz
tv, xv, λv = lyap_taylorinteg(lorenz!, x0, t0, tmax, 28, 1e-20, params; maxsteps=2000000);
nothing # hide
```
Now, the integration is obtained exploiting `lorenz_jac!`:
```@example lorenz
tv_, xv_, λv_ = lyap_taylorinteg(lorenz!, x0, t0, tmax, 28, 1e-20, params, lorenz_jac!; maxsteps=2000000);
nothing # hide
```
In terms of performance the second method is about ~50% faster than the first.

We check the consistency of the orbits computed by the two methods:
```@example lorenz
tv == tv_, xv == xv_, λv == λv_
```

As mentioned above, a more subtle check is related to the fact that the trace
of the Jacobian is constant in time, which must coincide with the sum
of all Lyapunov exponents. Using its initial value `lorenztr`, we compare it
with the final Lyapunov exponents of the computation, and obtain
```@example lorenz
sum(λv[end,:]) ≈ lorenztr, sum(λv_[end,:]) ≈ lorenztr, sum(λv[end,:]) ≈ sum(λv_[end,:])
```

Above we checked the approximate equality; we now show that the
relative error is quite small and comparable with the local machine epsilon
value around `lorenztr`:
```@example lorenz
abs(sum(λv[end,:])/lorenztr - 1), abs(sum(λv_[end,:])/lorenztr - 1), eps(lorenztr)
```
Therefore, the numerical error is dominated by roundoff errors in the floating
point arithmetic of the integration.
We will now proceed to plot our results. First, we plot Lorenz attractor in
phase space
```@example lorenz
using Plots
plot(xv[:,1], xv[:,2], xv[:,3], leg=false)
```

We display now the Lyapunov exponents as a function of time:
```@example lorenz
using Plots
nothing # hide
plot(tv, λv[:,1], label="L_1", legend=:right)
plot!(tv, λv[:,2], label="L_2")
plot!(tv, λv[:,3], label="L_3")
xlabel!("time")
ylabel!("L_i, i=1,2,3")
title!("Lyapunov exponents vs time")
```
This plot shows that the calculation of the Lyapunov exponents
has converged.
