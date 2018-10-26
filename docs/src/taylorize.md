# [Optimizing: `@taylorize`](@id taylorize)

Here, we describe the specific use of the macro [`@taylorize`](@ref), which
parses the functions containing the ODEs to be integrated, allowing to speed up
[`taylorinteg`](@ref) and [`lyap_taylorinteg`](@ref).

!!! warning
    The macro [`@taylorize`](@ref) is still in an experimental phase;
    be cautious of the resulting integration, which has to be tested
    carefully.


In order to explain what the macro does, we shall use as example the
[mathematical pendulum](@ref pendulum). First, we carry out the integration
as we did [before](@ref pendulum).
```@example taylorize
using TaylorIntegration

function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end

t0 = 0.0
tf = 100.0
q0 = [1.3, 0.0]

t1, x1 = taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500); # warm-up run
e1 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500);
e1
```

The actual number of methods defined for `TaylorIntegration.jetcoeffs!`
initially is 2.
```@example taylorize
length(methods(TaylorIntegration.jetcoeffs!)) == 2 # initial value
```
This number is maybe increased (new methods are added) when the macro,
or some of the internal functions around it, are used without an error.

The macro [`@taylorize`](@ref) is intended to be used in front of the function
that implements the equations of motion. The macro does the following: it
first parses the actual function as it is, so the integration can be computed
using `taylorinteg`, as above. It then creates a new method of
[`TaylorIntegration.jetcoeffs!`](@ref), which is specialized
(through `Val`) on the specific function parsed.

To see this, we use the macro on the function `pendulum!` as we wrote it above.
```@example taylorize
@taylorize function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end

methods(TaylorIntegration.jetcoeffs!)

methods(TaylorIntegration.jetcoeffs!)
```

We see that there is only one method of `pendulum!`, and
there is a new method of `TaylorIntegration.jetcoeffs!`, whose signature appears
in this documentation as `Val{Main.ex-taylorize.pendulum!}`; it is
an specialized version on the function `pendulum!`. This method
is selected internally by default when calling [`taylorinteg`](@ref) or
[`lyap_taylorinteg`](@ref) if it exists; to use the direct integration
illustrated above, the keyword argument `parse_eqs` has to be set to `false`.

Now we carry out the integration using the specialized method; note that we
use the same instruction used before.

```@example taylorize
t2, x2 = taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500); # warm-up run
e2 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500);
e2
```

We note the striking difference in the performance:
```@example taylorize
e1/e2
```

The obvious question is whether both integrations yield the same results.
```@example taylorize
t1 == t2 && x1 == x2
```

The speed-up obtained comes from the design of the new (specialized) method of
`TaylorIntegration.jetcoeffs!`: it avoids some repeated allocations incurred
by the direct implementation. This is achieved by knowing the specific AST of
the function that is integrated, which is walked through and *translated* into
the actual implementation, using directly the low-level functions defined in
`TaylorSeries.jl`.
For this, we use [Espresso.jl](https://github.com/dfdx/Espresso.jl).

For the function `pendulum!`, the new method is somewhat similar to:
```@example taylorize
ex = :(function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end)

TaylorIntegration._make_parsed_jetcoeffs(ex)
```
More complicated functions become difficult to read. From here, one can
further optimize this method manually.

Finally, in order to allow to opt-out from using the specialized method
created by [`@taylorize`](@ref), [`taylorinteg`](@ref) and
[`lyap_taylorinteg`](@ref) recognize the keyword argument `parse_eqs`;
setting it to `false` imposes using the standard method.
```@example taylorize
taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500, parse_eqs=false); # warm-up run

e3 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 28, 1e-20, maxsteps=1500, parse_eqs=false);
```
```@example taylorize
e1/e3
```

The macro parsing is somewhat complicated, and consequently, limited.
Broadcasting does not work, and its use is not compatible when used from
`DifferentialEquations.jl`. If the function is not recognized by [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl),
it is simply copied. The heuristics about vectors may not work for specific
cases. Please report any problems.
