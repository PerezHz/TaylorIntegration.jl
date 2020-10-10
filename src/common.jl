using DiffEqBase, OrdinaryDiffEq

import DiffEqBase: ODEProblem, solve, ODE_DEFAULT_NORM, @..
import OrdinaryDiffEq: OrdinaryDiffEqAdaptiveAlgorithm,
OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache,
alg_order, alg_cache, initialize!, perform_step!, @muladd, @unpack,
constvalue, @cache, tuple, stepsize_controller!, isfsal

abstract type TaylorAlgorithm <: OrdinaryDiffEqAdaptiveAlgorithm end
struct TaylorMethod <: TaylorAlgorithm
    order::Int
end

alg_order(alg::TaylorMethod) = alg.order

# Regarding `isfsal`, thought about setting isfsal to false, but low order
# interpolation may be helpful (since Taylor interpolation is memory-intensive),
# so let's set isfsal to true for now, which is the default, so no method
# definition needed here, so we in principle should support FSAL
# isfsal(::TaylorMethod) = true # see discussion above

TaylorMethod() = error("Maximum order must be specified for the TaylorMethod method")

export TaylorMethod

# overload DiffEqBase.ODE_DEFAULT_NORM for Taylor1 arrays
ODE_DEFAULT_NORM(x::AbstractArray{Taylor1{T}, N},y) where {T<:Number, N} = norm(x, Inf)

### cache stuff
struct TaylorMethodCache{uType,rateType,tTType,uTType} <: OrdinaryDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  k::rateType
  fsalfirst::rateType
  tT::tTType
  uT::uTType
  duT::uTType
  uauxT::uTType
  parse_eqs::Bool
end

full_cache(c::TaylorMethodCache) = begin
  tuple(c.u, c.uprev, c.tmp, c.k, c.fsalfirst, c.tT, c.uT, c.duT, c.uauxT, c.parse_eqs)
end

# @show macroexpand(@__MODULE__, :(@cache struct TaylorMethodCache{uType,rateType,tTType,uTType} <: OrdinaryDiffEqMutableCache
#   u::uType
#   uprev::uType
#   tmp::uType
#   k::rateType
#   fsalfirst::rateType
#   tT::tTType
#   uT::uTType
#   duT::uTType
#   uauxT::uTType
#   parse_eqs::Bool
# end) )

struct TaylorMethodConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(alg::TaylorMethod,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true})
  @show "alg_cache"
  TaylorMethodCache(
    u,
    u,
    zero(u),
    zero(rate_prototype),
    zero(rate_prototype),
    Taylor1(t, alg.order),
    Taylor1.(u, alg.order),
    zero.(Taylor1.(u, alg.order)),
    zero.(Taylor1.(u, alg.order)),
    true
    )
end

alg_cache(alg::TaylorMethod,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{false}) = TaylorMethodConstantCache()

function initialize!(integrator,cache::TaylorMethodConstantCache)
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
  integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
  integrator.destats.nf += 1

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator,cache::TaylorMethodConstantCache,repeat_step=false)
  @unpack t,dt,uprev,f,p = integrator
  @muladd u = @.. uprev + dt*integrator.fsalfirst
  k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
  integrator.destats.nf += 1
  integrator.fsallast = k
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.u = u
end

function initialize!(integrator, cache::TaylorMethodCache)
  @unpack u, t, f, p = integrator
  @unpack k, fsalfirst, tT, uT, duT, uauxT, parse_eqs = cache
  tT = Taylor1(t, integrator.alg.order)
  uT = Taylor1.(u, tT.order)
  duT = similar(uT)
  uauxT = similar(uT)
  parse_eqs = _determine_parsing!(parse_eqs, f.f, tT, uT, duT, p)
  # __jetcoeffs!(Val(parse_eqs), f.f, tT, uT, duT, uauxT, p)
  # integrator.dtpropose = stepsize(uT, integrator.opts.abstol)
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k
  f(integrator.fsalfirst, integrator.uprev, p, integrator.t) # For the interpolation, needs k at the updated point
  integrator.destats.nf += 1
end

function perform_step!(integrator, cache::TaylorMethodCache, repeat_step=false)
  @unpack t, dt, uprev, u, f, p = integrator
  @unpack tT, uT, duT, uauxT, parse_eqs = cache
  # integrator.dtpropose = stepsize(uT, integrator.opts.abstol)
  # @show uT
  tT[0] = t
  for i in eachindex(u)
    @inbounds uT[i][0] = u[i]
  end
  __jetcoeffs!(Val(parse_eqs), f.f, tT, uT, duT, uauxT, p)
  evaluate!(uT, dt, u)
  f(integrator.fsallast, u, p, t+dt) # For the interpolation, needs k at the updated point
  integrator.destats.nf += 1
end

# TODO: find how to actually implement time-stepping control using TaylorIntegration.stepsize
# stepsize_controller!(integrator,alg::TaylorMethod) = stepsize(integrator.cache.uT, integrator.opts.abstol)




# const warnkeywords =
# (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
# :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
# :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
# :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
# :isoutofdomain, :unstable_check,
# :calck, :progress, :timeseries_steps, :tstops, :dense)

# global warnlist = Set(warnkeywords)





# abstract type TaylorAlgorithm <: DiffEqBase.DEAlgorithm end
# struct TaylorMethod <: TaylorAlgorithm
#     order::Int
# end

# TaylorMethod() = error("Maximum order must be specified for the Taylor method")

# export TaylorMethod





# function DiffEqBase.solve(
#     prob::DiffEqBase.AbstractODEProblem{uType,tupType,isinplace},
#     alg::AlgType,
#     timeseries=[],ts=[],ks=[];
#     verbose=true,
#     saveat=eltype(tupType)[],
#     abstol = 1e-6,
#     save_start =  isempty(saveat) || saveat isa Number || prob.tspan[1] in saveat,
#     save_end   =  isempty(saveat) || saveat isa Number || prob.tspan[2] in saveat,
#     timeseries_errors=true, maxiters = 1_000_000,
#     callback=nothing, kwargs...) where
#         {uType, tupType, isinplace, AlgType <: TaylorAlgorithm}

#     tType = eltype(tupType)

#     if verbose
#         warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
#         warned && warn_compat()
#     end

#     if haskey(prob.kwargs, :callback) || haskey(kwargs, :callback) || !isa(callback, Nothing)
#         error("TaylorIntegration is not compatible with callbacks.")
#     end

#     sizeu = size(prob.u0)
#     f = prob.f.f

#     tspan = prob.tspan

#     _, saveat_vec_, _ = tstop_saveat_disc_handling(1,saveat,1,tspan)
#     saveat_vec = sort(saveat_vec_.valtree)

#     if !isempty(saveat) && save_end && saveat_vec[end] != tspan[2]
#         push!(saveat_vec,tspan[2])
#     end
#     if !isempty(saveat) && saveat_vec[1] != tspan[1]
#         prepend!(saveat_vec,tspan[1])
#     end

#     if !isinplace && typeof(prob.u0) <: Vector{Float64}
#         f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
#         if isempty(saveat_vec)
#             t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         else
#             t = saveat_vec
#             vectimeseries = taylorinteg(f!, prob.u0, t,
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         end

#     elseif !isinplace && typeof(prob.u0) <: AbstractArray
#         f! = (du, u, p, t) -> (du .= vec(f(reshape(u, sizeu), p, t)); 0)
#         if isempty(saveat_vec)
#             t, vectimeseries = taylorinteg(f!, prob.u0, prob.tspan[1], prob.tspan[2],
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         else
#             t = saveat_vec
#             vectimeseries = taylorinteg(f!, prob.u0, t,
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         end

#     elseif isinplace && typeof(prob.u0) <: AbstractArray{eltype(uType),2}
#         f! = (du, u, p, t) -> (
#                 dd = reshape(du, sizeu); uu = reshape(u, sizeu);
#                 f(dd, uu, p, t); u = vec(uu); du = vec(dd); 0)
#         u0 = vec(prob.u0)
#         if isempty(saveat_vec)
#             t, vectimeseries = taylorinteg(f!, u0, prob.tspan[1], prob.tspan[2],
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         else
#             t = saveat_vec
#             vectimeseries = taylorinteg(f!, prob.u0, t,
#                 alg.order, abstol, prob.p, maxsteps=maxiters, parse_eqs=false)
#         end
#     else
#         if haskey(kwargs, :parse_eqs)
#             parse_eqs = kwargs[:parse_eqs]
#         else
#             parse_eqs = true
#         end
#         if isempty(saveat_vec)
#             t, vectimeseries = taylorinteg(f, prob.u0, prob.tspan[1],
#                 prob.tspan[2], alg.order, abstol, prob.p, maxsteps=maxiters,
#                 parse_eqs=parse_eqs)
#         else
#             t = saveat_vec
#             vectimeseries = taylorinteg(f, prob.u0, t,
#                 alg.order, abstol, prob.p, maxsteps=maxiters,
#                 parse_eqs=parse_eqs)
#         end
#     end

#     if save_start
#       start_idx = 1
#       _t = t
#     else
#       start_idx = 2
#       _t = t[2:end]
#     end

#     if typeof(prob.u0) <: AbstractArray
#       _timeseries = Vector{uType}(undef, 0)
#       for i=start_idx:size(vectimeseries, 1)
#           push!(_timeseries, reshape(view(vectimeseries, i, :, )', sizeu))
#       end
#     else
#       _timeseries = vec(vectimeseries)[start_idx:end]
#     end

#     DiffEqBase.build_solution(prob,  alg, _t, _timeseries,
#         timeseries_errors = timeseries_errors, retcode = :Success)
# end
