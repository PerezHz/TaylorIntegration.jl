# [Optimizing: `@taylorize`](@id taylorize)

Here, we describe the specific use of the macro [`@taylorize`](@ref), which
parses the functions containing the ODEs to be integrated, allowing to speed up
[`taylorinteg`](@ref) and [`lyap_taylorinteg`](@ref).

!!! warning
    The macro [`@taylorize`](@ref) is still in an experimental phase;
    be cautious of the resulting integration, which has to be tested
    carefully.


## Basic usage

In order to explain what the macro does, we first run
a [simple example](@ref implementation_ex1) using
[`taylorinteg`](@ref) directly.
```@example taylorize
using TaylorIntegration
diffeq(t, x) = x^2
t1, x1 = taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150); # warm-up run
e1 = @elapsed taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150);
e1
```

We note that the actual number of methods defined for
`TaylorIntegration.jetcoeffs!` initially is 2:
```@example taylorize
length(methods(TaylorIntegration.jetcoeffs!)) == 2 # initial value
```
This number is not increased (new methods are added) as long as the macro,
or some of the internal functions around it, are not used.

The macro [`@taylorize`](@ref) is intended to be used in front of the function
that implements the equations of motion. The macro does the following: it
first parses the actual function as it is, so the method is defined and
can be used with `taylorinteg`. It then creates a new method of
[`TaylorIntegration.jetcoeffs!`](@ref), which is specialized
(through `Val`) on the specific function parsed.

To see this, we use the macro on the function `diffeq(t, x) = x^2`.
```@example taylorize
@taylorize diffeq(t, x) = x^2
methods(TaylorIntegration.jetcoeffs!)
```

There is a new method of `TaylorIntegration.jetcoeffs!`, whose signature appears
in this documentation as `Val{Main.ex-taylorize.diffeq}`; it is
an specialized version on the function `diffeq`. This method
is used (internally) by default when calling [`taylorinteg`](@ref) and
[`lyap_taylorinteg`](@ref) if it exists; to use the former integration the
keyword argument `parse_eqs` has to be set to `false`.

Now we carry out the integration using the specialized method; note that it
exactly the same instruction used before.

```@example taylorize
t2, x2 = taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150); # warm-up run
e2 = @elapsed taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150);
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

For the function `diffeq`, the new method is somewhat similar to:
```@example taylorize
TaylorIntegration._make_parsed_jetcoeffs( :(diffeq(t, x) = x^2) )
```
More complicated functions become difficult to read.

Finally, in order to allow to opt-out from using the specialized method
created by [`@taylorize`](@ref), [`taylorinteg`](@ref) and
[`lyap_taylorinteg`](@ref) recognize the keyword argument `parse_eqs`;
setting `parse_eqs = false` uses the standard method.
```@example taylorize
taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150, parse_eqs=false); # warm-up run
e3 = @elapsed taylorinteg(diffeq, 3.0, 0.0, 0.3, 28, 1e-20, maxsteps=150, parse_eqs=false);
e3/e1
```


## Some limitations

The macro parsing is somewhat complicated, and consequently, limited.

The limitations involve, on the one hand, functions that are defined in
[`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl),
and on other, some limitations related to walking the AST of the
function. Here, we outline some known limitations; please report any
problem or, even better, contribute with a solution.
