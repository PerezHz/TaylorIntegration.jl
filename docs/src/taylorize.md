# [Optimizing: `@taylorize`](@id taylorize)

Here, we describe the use of the macro [`@taylorize`](@ref), which
parses the functions containing the ODEs to be integrated, allowing to speed up
[`taylorinteg`](@ref) and [`lyap_taylorinteg`](@ref).

!!! warning
    The macro [`@taylorize`](@ref) is still in an experimental phase;
    be cautious of the resulting integration, which has to be tested
    carefully.

## [Some context and the idea](@id idea)

The way in which [`taylorinteg`](@ref) works by default is by calling
repeatedly the function where the ODEs of the problem are defined, in
order to compute the recurrence relations that are used to construct
the Taylor expansion of the solution. This is done for each order of
the series in [`TaylorIntegration.jetcoeffs!`](@ref). These computations are
not optimized: they waste memory due to allocations of some temporary
arrays, and perform some operations whose result has been previously
computed.

Here we describe one way to optimize this: The idea is to replace the
default method of [`TaylorIntegration.jetcoeffs!`](@ref) by another (with the
same name) which is called by dispatch, that in principle performs better.
The new method is constructed specifically for the actual function
defining the equations of motion by parsing its expression; the new
function performs in principle *exactly* the same operations, but avoids
the extra allocations and the repetition of some operations.


## An example

In order to explain how the macro works, we shall use as an example the
[mathematical pendulum](@ref pendulum). First, we carry out the integration
using the default method, as described [before](@ref pendulum).

```@example taylorize
using TaylorIntegration

function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return dx
end

# Initial time (t0), final time (tf) and initial condition (q0)
t0 = 0.0
tf = 100.0
q0 = [1.3, 0.0]

# The actual integration
t1, x1 = taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500); # warm-up run
e1 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500);
e1
```

We note that the initial number of methods defined for
`TaylorIntegration.jetcoeffs!` is 2.
```@example taylorize
length(methods(TaylorIntegration.jetcoeffs!)) == 2 # initial value
```
Using `@taylorize` will increase this number by creating a new method.

The macro [`@taylorize`](@ref) is intended to be used in front of the function
that implements the equations of motion. The macro does the following: it
first parses the actual function as it is, so the integration can be computed
using [`taylorinteg`](@ref) as above, by explicitly using the keyword
argument `parse_eqs=false`. It then creates and evaluates a new method of
[`TaylorIntegration.jetcoeffs!`](@ref), which is the specialized method
(through `Val`) on the specific function passed to the macro.

```@example taylorize
@taylorize function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return dx
end

println(methods(pendulum!))
```

```@example taylorize
println(methods(TaylorIntegration.jetcoeffs!))
```

We see that there is only one method of `pendulum!`, and
there is a new method of `TaylorIntegration.jetcoeffs!`, whose signature appears
in this documentation as `Val{Main.ex-taylorize.pendulum!}`; it is
an specialized version for the function `pendulum!` (with some extra information
about the module where the function was created). This method
is selected internally if it exists (default), exploiting dispatch, when
calling [`taylorinteg`](@ref) or [`lyap_taylorinteg`](@ref); to integrate
using the hard-coded method of [`TaylorIntegration.jetcoeffs!`](@ref) of the
integration above, the keyword argument `parse_eqs` has to be set to `false`.

Now we carry out the integration using the specialized method; note that we
use the same instruction as above.

```@example taylorize
t2, x2 = taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500); # warm-up run
e2 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500);
e2
```

We note the difference in the performance:
```@example taylorize
e1/e2
```

We can check that both integrations yield the same results.
```@example taylorize
t1 == t2 && x1 == x2
```

As stated above, in order to allow to opt-out from using the specialized method
created by [`@taylorize`](@ref), [`taylorinteg`](@ref) and
[`lyap_taylorinteg`](@ref) recognize the keyword argument `parse_eqs`;
setting it to `false` imposes using the standard method.
```@example taylorize
taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500, parse_eqs=false); # warm-up run

e3 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 25, 1e-20, maxsteps=1500, parse_eqs=false);

e1/e3
```

We now illustrate the possibility of exploiting the macro
when using `TaylorIntegration.jl` from `DifferentialEquations.jl`.
```@example taylorize
using DiffEqBase

prob = ODEProblem(pendulum!, q0, (t0, tf), nothing) # no parameters
solT = solve(prob, TaylorMethod(25), abstol=1e-20, parse_eqs=true); # warm-up run
e4 = @elapsed solve(prob, TaylorMethod(25), abstol=1e-20, parse_eqs=true);

e1/e4
```
Note that there is a marginal cost of using `solve` in comparison
with `taylorinteg`.

The speed-up obtained comes from the design of the new (specialized) method of
`TaylorIntegration.jetcoeffs!` as described [above](@ref idea): it avoids some
allocations and some repeated computations. This is achieved by knowing the
specific AST of the function of the ODEs integrated, which is walked
through and *translated* into the actual implementation, where some
required auxiliary arrays are created and the low-level functions defined in
`TaylorSeries.jl` are used.
For this, we heavily rely on [`Espresso.jl`](https://github.com/dfdx/Espresso.jl) and
some metaprogramming; we thank Andrei Zhabinski for his help and comments.

The new `jetcoeffs!` method can be inspected by constructing the expression
corresponding to the function, and using
[`TaylorIntegration._make_parsed_jetcoeffs`](@ref):

```@example taylorize
ex = :(function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return dx
end)

new_ex = TaylorIntegration._make_parsed_jetcoeffs(ex)
```

This function has a similar structure as the hard-coded method of
`TaylorIntegration.jetcoeffs!`, but uses low-level functions in `TaylorSeries`
(e.g., `sincos!` above) and explicitly allocates the needed temporary arrays.
More complex functions become easily very difficult to read. Note that,
if necessary, one can further optimize `new_ex` manually.


## Limitations and some advices

The construction of the internal function obtained by using
[`@taylorize`](@ref) is somewhat complicated and limited. Here we
list some limitations and advices.

- It is useful to have expressions which involve two arguments at most, which
  imposes the proper use of parenthesis: For example, `res = a+b+c` should be
  written as `res = (a+b)+c`.

- Updating operators such as `+=`, `*=`, etc., are not supported. For
  example, the expression `x += y` is not recognized by `@taylorize`. Likewise,
  expressions such as `x = x+y` are not supported by `@taylorize` and should be
  substituted by equivalent expressions; e.g. `z = x+y; x = z`.

- The macro allows to use array declarations through `Array`, but other ways
  (e.g. `similar`) are not yet implemented.

- Avoid using variables prefixed by an underscore, in particular `_T`, `_S` and
  `_N`; using them may lead to name collisions with some internal variables.

- Broadcasting is not recognized by `@taylorize`.

- The macro may be used in combination with the [common interface with
  `DifferentialEquations.jl`](@ref diffeqinterface), for functions using the
  `(du, u, p, t)` in-place form, as we showed above. Other extensions allowed by
  `DifferentialEquations` may not be able to exploit it.

- `if-else` blocks are recognized in its long form, but short-circuit
  conditional operators (`&&` and `||`) are not.

- Expressions which correspond to function calls (so the `head` field is
  `:call`) which are not recognized by the parser are simply copied. The
  heuristics used, specially for vectors, may not work for all cases.

- Use `local` for internal parameters (simple constant values); this improves
  performance. Do not use it if the variable is Taylor expanded.

- `@taylorize` supports multi-threading via `Threads.@threads`. **WARNING**:
  this feature is experimental. Since thread-safety depends on the definition
  of each ODE, we cannot guarantee the resulting code to be thread-safe in
  advance. The user should check the resulting code to ensure that it is indeed
  thread-safe. For more information about multi-threading, the reader is
  referred to the [Julia documentation](https://docs.julialang.org/en/v1/manual/parallel-computing#man-multithreading-1).

It is recommended to skim `test/taylorize.jl`, which implements different
cases.

Please report any problems you may encounter.
