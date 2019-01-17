# [Optimizing: `@taylorize`](@id taylorize)

Here, we describe the use of the macro [`@taylorize`](@ref), which
parses the functions containing the ODEs to be integrated, allowing to speed up
[`taylorinteg`](@ref) and [`lyap_taylorinteg`](@ref).

!!! warning
    [`@taylorize`](@ref) is still in an experimental phase;
    be cautious of the resulting integration, which has to be tested
    carefully.


## An example

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
`@taylorize` will increase this number, that is, new methods will be created.

The macro [`@taylorize`](@ref) is intended to be used in front of the function
that implements the equations of motion. The macro does the following: it
first parses the actual function as it is, so the integration can be computed
using `taylorinteg`, as above (or with the keyword argument `parse_eqs=false`).
It then creates a new method of [`TaylorIntegration.jetcoeffs!`](@ref), which
is specialized method (through `Val`) on the specific function parsed.

To see this, we use the macro on the function `pendulum!` as we wrote it above.
```@example taylorize
@taylorize function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end

methods(pendulum!)
```

```@example taylorize
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
use the same instruction as before.

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
`TaylorIntegration.jetcoeffs!`: it avoids some repeated computations incurred
by the direct implementation. This is achieved by knowing the specific AST of
the function that is integrated, which is walked through and *translated* into
the actual implementation, using directly the low-level functions defined in
`TaylorSeries.jl`.
For this, we rely on [Espresso.jl](https://github.com/dfdx/Espresso.jl) and
some metaprogramming.

For the function `pendulum!`, the new method can be obtained as:
```@example taylorize
ex = :(function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end)

new_ex = TaylorIntegration._make_parsed_jetcoeffs(ex)
```
The complexity of the functions increases the difficult to read them. Note that
one can further optimize `new_ex` manually.

As stated above, in order to allow to opt-out from using the specialized method
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

## Limitations

The macro parsing is somewhat complicated and somewhat limited. It does allow
to use array declarations through `Array`, but other ways (such as `similar`)
are not yet implemented; avoid using `_T` and `_S` as type variables
in the array definitions, since these names are used to define the
parameterized types of the function as well as proper initialization of
some temporary arrays. Broadcasting is not compatible with `@taylorize`,
and it can't be exploited from `DifferentialEquations.jl` interface. `if-else` blocks
are recognized, but short-circuit conditional operators (`&&` and `||`)
are not. Function calls which are not recognized by
[`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl),
are simply copied. The heuristics for vectors may not work for all
cases.

We recommend to have a look on `test/taylorize.jl`, which implements different
cases, including uses of `local` for internal parameters. Please report any
problems you may encounter.
