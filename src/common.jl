using DiffEqBase

import DiffEqBase: ODEProblem, solve

const warnkeywords =
    (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
     :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
     :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
     :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
     :maxiters, :isoutofdomain, :unstable_check,
     :calck, :progress, :timeseries_steps, :tstops, :saveat, :dense)

global warnlist = Set(warnkeywords)



abstract type TaylorAlgorithm <: DiffEqBase.DEAlgorithm end
struct TaylorMethod <: TaylorAlgorithm
    order::Int
end

TaylorMethod() = error("Maximum order must be specified for the Taylor method")

export TaylorMethod

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem{uType,tType,isinplace},
    alg::AlgType,
    timeseries=[],ts=[],ks=[];
    verbose=true, abstol = 1e-6, save_start = true,
    timeseries_errors=true, maxiters = 1_000_000,
    kwargs...) where
        {uType, tType, isinplace, AlgType <: TaylorAlgorithm}

    if verbose
        warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
        warned && warn_compat()
    end

    if haskey(prob.kwargs, :callback) || haskey(kwargs, :callback)
        error("TaylorIntegration is not compatible with callbacks.")
    end

    sizeu = size(prob.u0)
    f = prob.f

    if !isinplace && typeof(prob.u0) <: Vector{Float64}
        f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
        t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
            alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)

    elseif !isinplace && typeof(prob.u0) <: AbstractArray
        f! = (du, u, p, t) -> (du .= vec(f(reshape(u, sizeu), p, t)); 0)
        t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
            alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)

    elseif isinplace && typeof(prob.u0) <: AbstractArray{eltype(uType),2}
        f! = (du, u, p, t) -> (
                dd = reshape(du, sizeu); uu = reshape(u, sizeu);
                f(dd, uu, p, t); u = vec(uu); du = vec(dd); 0)
        u0 = vec(prob.u0)
        t, vectimeseries = taylorinteg(f!, u0, prob.tspan[1], prob.tspan[2],
            alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)

    else
        if haskey(kwargs, :parse_eqs)
            parse_eqs = kwargs[:parse_eqs]
        else
            parse_eqs = true
        end
        t, vectimeseries = taylorinteg(prob.f.f, prob.u0, prob.tspan[1],
            prob.tspan[2], alg.order, abstol, prob.p, maxsteps=maxiters,
            parse_eqs=parse_eqs)

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
      _timeseries = vec(vectimeseries)
    end

    DiffEqBase.build_solution(prob,  alg, _t, _timeseries,
        timeseries_errors = timeseries_errors, retcode = :Success)
end
