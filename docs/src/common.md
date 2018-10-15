# [Interoperability between TaylorIntegration.jl and the `JuliaDiffEq` package ecosystem](@id diffeqinterface)

Below, we show an example of usage of `TaylorIntegration.jl` together with
some packages from the `JuliaDiffEqs` organization. Below, we use
`ParameterizedFunctions.jl` to define an ODE with human-readable mathematical
notation. Also, we use `OrdinaryDiffEqs.jl`, in order to compare the performance
and accuracy of `TaylorIntegration.jl` with respect to high-accuracy methods for
non-stiff problems.

The problem we will integrate in this example is the Planar, Circular Restricted
Three-Body Problem (PCR3BP):

```@example common
using ParameterizedFunctions
f = @ode_def PCR3BP begin
    dx = px + y
    dy = py - x
    dpx = - (1-μ)*(x-μ)*((x-μ)^2+y^2)^-1.5 - μ*(x+1-μ)*((x+1-μ)^2+y^2)^-1.5 + py
    dpy = - (1-μ)*y    *((x-μ)^2+y^2)^-1.5 - μ*y      *((x+1-μ)^2+y^2)^-1.5 - px
end μ
nothing # hide
```
The mass parameter $\mu = m/(m+M)$ has a value:
```@example common
μ = 0.01
nothing # hide
```

```@example common
d_prim(x, y, px, py) = sqrt((x-μ)^2+y^2)
d_sec(x, y, px, py) = sqrt((x+1-μ)^2+y^2)
d_prim(x) = d_prim(x...)
d_sec(x) = d_sec(x...)
V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
H(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)
H(x) = H(x...)
ZVC(x) =  -x^2/2 + V(x, zero(x)) # zero-velocity curve
nothing # hide
```

```@example common
using Plots
plot(ZVC, -2:0.001:2)
ylims!(-3, -1)
```

```@example common
function py!(u0, E0)
    @assert u0[2] == zero(eltype(u0)) "u0[2] has to be equal to zero"
    @assert u0[3] == zero(eltype(u0)) "u0[3] has to be equal to zero"
    u0[4] = u0[1] + sqrt( u0[1]^2-2( V(u0[1], u0[2])-E0 ) )
    nothing
end
nothing # hide
```

```@example common
u0 = [-0.8, 0.0, 0.0, 0.0]
E0 = -1.5772
py!(u0, E0)
```

```@example common
# check that the energy of the initial condition is indeed equal to E0
H(u0) == E0, H(u0) - E0
```

```@example common
tspan = (0.0, 1000.0)
p = [μ]
using TaylorIntegration
prob = ODEProblem(f, u0, tspan, p)
```

```@example common
@time solT = solve(prob, TaylorMethod(25), abstol=1e-20);
```

```@example common
using OrdinaryDiffEq
```

```@example common
@time solV = solve(prob, Vern9());
```

```@example common
E0 = H(u0)
ET = H.(solT.u)
EV = H.(solV.u)
δET = ET .- E0
δEV = EV .- E0
nothing # hide
```

```@example common
plot(solT, vars=(1, 2))
scatter!([μ, -1+μ], [0,0], leg=false)
xlims!(-1+μ-0.2, 1+μ+0.2)
```

```@example common
plot(solV, vars=(1, 2))
scatter!([μ, -1+μ], [0,0], leg=false)
xlims!(-1+μ-0.2, 1+μ+0.2)
```

```@example common
plot(solT.t, abs.(δET), yscale=:log10)
plot!(solV.t, abs.(δEV))
ylims!(10^-18,10^4)
```

```@example common
d1 = d_prim.(solT.u) # distance to primary
d2 = d_sec.(solT.u); #distance to secondary
```

```@example common
plot(solT.t, d1, yscale=:log10, label="d(Prim)")
plot!(solT.t, d2, label="d(Sec)")
```
