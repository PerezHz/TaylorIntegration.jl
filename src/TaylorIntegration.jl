# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using TaylorSeries

import TaylorSeries: evaluate

export  taylorinteg,
        # integrate!,
        # integrate_k!,
        taylorstep!,
        evaluate,
        stepsize,
        jetcoeffs!

        # taylor_integrator_v2!,
        # taylor_one_step_v2!,
        # taylor_integrator_log

include("taylor_integration_methods.jl")

end #module
