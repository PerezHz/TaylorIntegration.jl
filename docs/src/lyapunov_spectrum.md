# [Lyapunov spectrum](@id lyap)

Here we describe the background of Lyapunov spectra computations in
`TaylorIntegration.jl`.

Implementation of Lyapunov spectra computations in `TaylorIntegration.jl`
follow the numerical method of Benettin et al. [1, 2], which itself is based on
Oseledet's multiplicative ergodic theorem [3].



## References

[1] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980a, Meccanica, 15, 9
[2] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980b, Meccanica, 15, 21
[3] Oseledets V. I., 1968, Trudy Moskovskogo Matematicheskogo Obshchestva, 19, 179