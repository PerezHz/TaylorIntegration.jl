# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using TaylorSeries

import Compat.view

export  taylorinteg

include("integration_methods.jl")

end #module
