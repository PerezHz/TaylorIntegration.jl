# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using TaylorSeries

import Compat.view

export  taylorinteg

include("explicitode.jl")

end #module
