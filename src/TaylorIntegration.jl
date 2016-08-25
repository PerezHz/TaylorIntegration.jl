# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using TaylorSeries

import TaylorSeries: evaluate

export  taylorinteg,
        taylorstep!,
        evaluate,
        stepsize,
        jetcoeffs!

include("taylor_integration_methods.jl")

end #module
