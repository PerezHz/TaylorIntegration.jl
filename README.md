# TaylorIntegration.jl

ODE integration using Taylor's method in [Julia](http://julialang.org).

Build status:
[![travis][travis-img]](https://travis-ci.org/PerezHz/TaylorIntegration.jl)
[![appveyor][appveyor-img]](https://ci.appveyor.com/project/PerezHz/taylorintegration-jl/branch/master)

Coverage:
[![Coverage Status](https://coveralls.io/repos/github/PerezHz/TaylorIntegration.jl/badge.svg?branch=master)](https://coveralls.io/github/PerezHz/TaylorIntegration.jl?branch=master) [![codecov](https://codecov.io/gh/PerezHz/TaylorIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PerezHz/TaylorIntegration.jl)

[travis-img]: https://img.shields.io/travis/PerezHz/TaylorIntegration.jl/master.svg?label=Linux+/+macOS
[appveyor-img]: https://img.shields.io/travis/PerezHz/TaylorIntegration.jl/master.svg?label=Windows
[coveralls-img]: https://img.shields.io/travis/PerezHz/TaylorIntegration.jl/master.svg?label=coveralls
[codecov-img]: https://img.shields.io/travis/PerezHz/TaylorIntegration.jl/master.svg?label=codecov

Documentation:
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://PerezHz.github.io/TaylorIntegration.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://PerezHz.github.io/TaylorIntegration.jl/stable)

DOI (Zenodo):
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3241532.svg)](https://doi.org/10.5281/zenodo.3241532)

## Authors

- [Jorge A. Pérez](https://www.linkedin.com/in/perezhz),
Instituto de Ciencias Físicas, Universidad Nacional Autónoma de México (UNAM)
- [Luis Benet](http://www.cicc.unam.mx/~benet/),
Instituto de Ciencias Físicas, Universidad Nacional Autónoma de México (UNAM)

Comments, suggestions, and improvements are welcome and appreciated.

## Examples

> *NOTE: Version 0.5.0 includes breaking changes related with the
form of the function that defines the equations of motion (which follows now the convention of [`DifferentialEquations.jl`](http://docs.juliadiffeq.org/stable/tutorials/ode_example.html)). Then, code that used to work in previous versions has to be amended.*

+ [x'=x^2](http://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/x-dot-equals-x-squared.ipynb)
+ [Kepler problem](http://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/Kepler-problem.ipynb)
+ [High-order polynomial approximations to special functions and integral transforms](http://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/Polynomial-approx-special-functions-integral-transforms.ipynb)
+ [Damped, driven linear oscillator](http://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/Damped-driven-linear-oscillator.ipynb)
+ [The Lyapunov spectrum of the Lorenz system](http://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/Lorenz-Lyapunov-spectrum.ipynb)
+ TaylorIntegration @ JuliaCon 2017 [(slides)](http://nbviewer.jupyter.org/format/slides/github/PerezHz/TaylorIntegration.jl/blob/master/examples/JuliaCon2017/TaylorIntegration_JuliaCon.ipynb)

## License

`TaylorIntegration` is licensed under the [MIT "Expat" license](./LICENSE.md).

## Acknowledgments

We acknowledge financial support from DGAPA-PAPIIT grants IG-100616.
