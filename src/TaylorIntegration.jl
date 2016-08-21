module TaylorIntegration

using TaylorSeries

export  #taylor_integrator,
        integrate!,
        integrate_k!,
        iterate!,
        propagate,
        stepsize, stepsizeall,
        differentiate!

        # taylor_integrator_v2!,
        # taylor_one_step_v2!,
        # taylor_integrator_log


include("taylor_integration_methods.jl")

end #module
