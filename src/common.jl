using DiffEqBase

import DiffEqBase: ODEProblem, solve
import OrdinaryDiffEq: tstop_saveat_disc_handling

const warnkeywords =
    (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
     :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
     :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
     :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
     :maxiters, :isoutofdomain, :unstable_check,
     :calck, :progress, :timeseries_steps, :tstops, :dense)

global warnlist = Set(warnkeywords)



abstract type TaylorAlgorithm <: DiffEqBase.DEAlgorithm end
struct TaylorMethod <: TaylorAlgorithm
    order::Int
end

TaylorMethod() = error("Maximum order must be specified for the Taylor method")

export TaylorMethod

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem{uType,tupType,isinplace},
    alg::AlgType,
    timeseries=[],ts=[],ks=[];
    verbose=true,
    saveat=eltype(tupType)[],
    abstol = 1e-6,
    save_start =  isempty(saveat) || saveat isa Number || prob.tspan[1] in saveat,
    save_end   =  isempty(saveat) || saveat isa Number || prob.tspan[2] in saveat,
    timeseries_errors=true, maxiters = 1_000_000,
    callback=nothing, kwargs...) where
        {uType, tupType, isinplace, AlgType <: TaylorAlgorithm}

    tType = eltype(tupType)

    if verbose
        warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
        warned && warn_compat()
    end

    if haskey(prob.kwargs, :callback) || haskey(kwargs, :callback) || !isa(callback, Nothing)
        error("TaylorIntegration is not compatible with callbacks.")
    end

    sizeu = size(prob.u0)
    f = prob.f.f

    tspan = prob.tspan

    _, saveat_vec_, _ = tstop_saveat_disc_handling(1,saveat,1,tspan)
    saveat_vec = sort(saveat_vec_.valtree)

    if !isempty(saveat) && save_end && saveat_vec[end] != tspan[2]
        push!(saveat_vec,tspan[2])
    end
    if !isempty(saveat) && saveat_vec[1] != tspan[1]
        prepend!(saveat_vec,tspan[1])
    end

    if !isinplace && typeof(prob.u0) <: Vector{Float64}
        f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
        if isempty(saveat_vec)
            t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        else
            t = saveat_vec
            vectimeseries = taylorinteg(f!, prob.u0, t,
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        end

    elseif !isinplace && typeof(prob.u0) <: AbstractArray
        f! = (du, u, p, t) -> (du .= vec(f(reshape(u, sizeu), p, t)); 0)
        if isempty(saveat_vec)
            t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        else
            t = saveat_vec
            vectimeseries = taylorinteg(f!, prob.u0, t,
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        end

    elseif isinplace && typeof(prob.u0) <: AbstractArray{eltype(uType),2}
        f! = (du, u, p, t) -> (
                dd = reshape(du, sizeu); uu = reshape(u, sizeu);
                f(dd, uu, p, t); u = vec(uu); du = vec(dd); 0)
        u0 = vec(prob.u0)
        if isempty(saveat_vec)
            t, vectimeseries = taylorinteg(f!, u0, prob.tspan[1], prob.tspan[2],
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        else
            t = saveat_vec
            vectimeseries = taylorinteg(f!, prob.u0, t,
                alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
        end
    else
        if haskey(kwargs, :parse_eqs)
            parse_eqs = kwargs[:parse_eqs]
        else
            parse_eqs = true
        end
        if isempty(saveat_vec)
            t, vectimeseries = taylorinteg(f, prob.u0, prob.tspan[1],
                prob.tspan[2], alg.order, abstol, prob.p, maxsteps=maxiters,
                parse_eqs=parse_eqs)
        else
            t = saveat_vec
            vectimeseries = taylorinteg(f, prob.u0, t,
                alg.order, abstol, prob.p, maxsteps=maxiters,
                parse_eqs=parse_eqs)
        end
    end

    if save_start
      start_idx = 1
      _t = t
    else
      start_idx = 2
      _t = t[2:end]
    end

    if typeof(prob.u0) <: AbstractArray
      _timeseries = Vector{uType}(undef, 0)
      for i=start_idx:size(vectimeseries, 1)
          push!(_timeseries, reshape(view(vectimeseries, i, :, )', sizeu))
      end
    else
      _timeseries = vec(vectimeseries)[start_idx:end]
    end

    DiffEqBase.build_solution(prob,  alg, _t, _timeseries,
        timeseries_errors = timeseries_errors, retcode = :Success)
end
