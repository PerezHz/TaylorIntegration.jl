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
    timeseries_errors=true, maxiters = 1000000,
    callback=nothing, kwargs...) where {uType, tType, isinplace, AlgType <: TaylorAlgorithm}

    if verbose
        warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
        warned && warn_compat()
    end

    if prob.callback != nothing || callback != nothing
        error("TaylorIntegration is not compatible with callbacks.")
    end

    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        u0 = vec(deepcopy(prob.u0))
    end

    sizeu = size(prob.u0)
    p = prob.p
    f = prob.f

    if !isinplace && (typeof(prob.u0)<:Vector{Float64} || typeof(prob.u0)<:Number)
        f! = (t, u, du) -> (du .= f(u, p, t); 0)
    elseif !isinplace && typeof(prob.u0)<:AbstractArray
        f! = (t, u, du) -> (du .= vec(f(reshape(u, sizeu), p, t)); 0)
    elseif typeof(prob.u0)<:Vector{Float64}
        f! = (t, u, du) -> f(du, u, p, t)
    else # Then it's an in-place function on an abstract array
        f! = (t, u, du) -> (f(reshape(du, sizeu), reshape(u, sizeu), p, t);
                            u = vec(u); du=vec(du); 0)
    end

    t,vectimeseries = taylorinteg(f!, u0, prob.tspan[1], prob.tspan[2], alg.order,
                                                      abstol, maxsteps=maxiters)

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
                   timeseries_errors = timeseries_errors,
                   retcode = :Success)
end
