# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries, DiffEqBase
using LinearAlgebra

const warnkeywords =
    (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
     :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
     :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
     :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
     :maxiters, :isoutofdomain, :unstable_check,
     :calck, :progress, :timeseries_steps, :tstops, :saveat, :dense)

function __init__()
    global warnlist = Set(warnkeywords)
end

export taylorinteg, lyap_taylorinteg, @taylorize

include("explicitode.jl")
include("lyapunovspectrum.jl")
include("common.jl")
include("rootfinding.jl")

include("parse_eqs.jl")

end #module
