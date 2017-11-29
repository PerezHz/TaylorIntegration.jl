# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using TaylorSeries

using Reexport
@reexport using DiffEqBase

const warnkeywords =
    (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
     :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
     :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
     :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
     :maxiters, :isoutofdomain, :unstable_check,
     :calck, :progress, :timeseries_steps, :tstops, :saveat, :dense)

function __init__()
    const global warnlist = Set(warnkeywords)
end

export taylorinteg, liap_taylorinteg

include("explicitode.jl")
include("liapunovspectrum.jl")
include("common.jl")

include("rootfinding.jl")

end #module
