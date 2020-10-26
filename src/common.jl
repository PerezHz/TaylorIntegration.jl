using DiffEqBase, OrdinaryDiffEq

import DiffEqBase: ODEProblem, solve, ODE_DEFAULT_NORM, @.., addsteps!

import OrdinaryDiffEq: OrdinaryDiffEqAdaptiveAlgorithm,
OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache,
alg_order, alg_cache, initialize!, perform_step!, @unpack,
@cache, stepsize_controller!, step_accept_controller!

# TODO: check which keywords work fine
const warnkeywords = (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
:save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
:dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
:qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
:isoutofdomain, :unstable_check,
:calck, :progress, :timeseries_steps, :dense)

global warnlist = Set(warnkeywords)



abstract type TaylorAlgorithm <: OrdinaryDiffEqAdaptiveAlgorithm end
struct TaylorMethod <: TaylorAlgorithm
    order::Int
end
struct _TaylorMethod <: TaylorAlgorithm
    order::Int
    parse_eqs::Bool
end

_TaylorMethod(order; parse_eqs=true) = _TaylorMethod(order, parse_eqs) # set `parse_eqs` to `true` by default

alg_order(alg::TaylorMethod) = alg.order
alg_order(alg::_TaylorMethod) = alg.order

# Regarding `isfsal`, thought about setting isfsal to false, but low order
# interpolation may be helpful (since Taylor interpolation is memory-intensive),
# so let's set isfsal to true for now, which is the default, so no method
# definition needed here, so we in principle should support FSAL
# isfsal(::TaylorMethod) = true # see discussion above

TaylorMethod() = error("Maximum order must be specified for the Taylor method")

export TaylorMethod

# overload DiffEqBase.ODE_DEFAULT_NORM for Taylor1 arrays
ODE_DEFAULT_NORM(x::AbstractArray{Taylor1{T}, N},y) where {T<:Number, N} = norm(x, Inf)

### cache stuff
struct TaylorMethodCache{uType, rateType, tTType, uTType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    tT::tTType
    uT::uTType
    duT::uTType
    uauxT::uTType
    parse_eqs::Ref{Bool}
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

struct TaylorMethodConstantCache{uTType} <: OrdinaryDiffEqConstantCache
    uT::uTType
    parse_eqs::Ref{Bool}
end

function alg_cache(alg::_TaylorMethod, u, rate_prototype, uEltypeNoUnits,
        uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p,
        calck,::Val{true})
    TaylorMethodCache(
        u,
        uprev,
        similar(u),
        zero(rate_prototype),
        zero(rate_prototype),
        Taylor1(t, alg.order),
        Taylor1.(u, alg.order),
        zero.(Taylor1.(u, alg.order)),
        zero.(Taylor1.(u, alg.order)),
        Ref(alg.parse_eqs)
        )
end

alg_cache(alg::_TaylorMethod, u, rate_prototype, uEltypeNoUnits,
    uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck,
    ::Val{false}) = TaylorMethodConstantCache(Taylor1(u, alg.order), Ref(alg.parse_eqs))

function initialize!(integrator, c::TaylorMethodConstantCache)
    @unpack u, t, f, p = integrator
    tT = Taylor1(typeof(t), integrator.alg.order)
    tT[0] = t
    c.uT .= Taylor1(u, tT.order)
    c.parse_eqs.x = _determine_parsing!(c.parse_eqs.x, f, tT, c.uT, p)
    __jetcoeffs!(Val(c.parse_eqs.x), f, tT, c.uT, p)
    # FSAL stuff
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.destats.nf += 1
    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator,cache::TaylorMethodConstantCache)
    @unpack u, t, dt, f, p = integrator
    tT = Taylor1(typeof(t), integrator.alg.order)
    tT[0] = t+dt
    u = evaluate(cache.uT, dt)
    cache.uT[0] = u
    __jetcoeffs!(Val(cache.parse_eqs.x), f, tT, cache.uT, p)
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
    tT .= Taylor1(typeof(t), integrator.alg.order)
    tT[0] = t
    uT .= Taylor1.(u, tT.order)
    duT .= zero.(Taylor1.(u, tT.order))
    uauxT .= similar(uT)
    parse_eqs.x = _determine_parsing!(parse_eqs.x, f, tT, uT, duT, p)
    __jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p)
    # FSAL for interpolation
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    integrator.kshortsize = 1
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    # integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t)
    integrator.fsalfirst = constant_term.(duT)
    integrator.destats.nf += 1
end

function perform_step!(integrator, cache::TaylorMethodCache)
    @unpack t, dt, u, f, p = integrator
    @unpack k, tT, uT, duT, uauxT, parse_eqs = cache
    evaluate!(uT, dt, u)
    tT[0] = t+dt
    for i in eachindex(u)
        @inbounds uT[i][0] = u[i]
        duT[i].coeffs .= zero(duT[i][0])
    end
    __jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p)
    k = constant_term.(duT) # For the interpolation, needs k at the updated point
    integrator.destats.nf += 1
end

stepsize_controller!(integrator,alg::_TaylorMethod) = stepsize(integrator.cache.uT, integrator.opts.abstol)
step_accept_controller!(integrator, alg::_TaylorMethod, q) = q

function DiffEqBase.solve(
        prob::DiffEqBase.AbstractODEProblem{uType, tupType, isinplace},
        alg::TaylorMethod, args...;
        verbose=true,
        kwargs...) where
        {uType, tupType, isinplace}

    if verbose
        warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
        warned && warn_compat()
    end

    sizeu = size(prob.u0)
    # This if block handles DynamicalODEProblems
    # credit to @SebastianM-C for coming up with this solution
    if prob.f isa DynamicalODEFunction
        _alg = _TaylorMethod(alg.order, parse_eqs = false)
        if isinplace
            f = (du,u,p,t) ->
            @inbounds begin
                dv1 = view(du, firstindex(du):lastindex(du)÷2)
                dv2 = view(du, lastindex(du)÷2+1:lastindex(du))
                v1 = view(u, firstindex(u):lastindex(u)÷2)
                v2 = view(u, lastindex(u)÷2+1:lastindex(u))
                prob.f.f1(dv1, v1, v2, p, t)
                prob.f.f2(dv2, v1, v2, p, t)
                return nothing
            end
        else
            f = (du,u,p,t) ->
            @inbounds begin
                dv1 = view(du, firstindex(du):lastindex(du)÷2)
                dv2 = view(du, lastindex(du)÷2+1:lastindex(du))
                v1 = view(u, firstindex(u):lastindex(u)÷2)
                v2 = view(u, lastindex(u)÷2+1:lastindex(u))
                dv1 = prob.f.f1(v1, v2, p, t)
                dv2 = prob.f.f2(v1, v2, p, t)
                return nothing
            end
        end
        _u0 = convert(Array{eltype(prob.u0)}, prob.u0)
        _prob = ODEProblem(f, _u0, prob.tspan, prob.p; prob.kwargs...)
        # DiffEqBase.solve(prob, _alg, args...; kwargs...)
        integrator = DiffEqBase.__init(_prob, _alg, args...; kwargs...)
    else
        f = prob.f
        if !isinplace && typeof(prob.u0) <: AbstractArray
            f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
            _alg = _TaylorMethod(alg.order, parse_eqs = false)
            prob.f = f!
        elseif haskey(kwargs, :parse_eqs)
            _alg = _TaylorMethod(alg.order, parse_eqs = kwargs[:parse_eqs])
        else
            _alg = _TaylorMethod(alg.order)
        end
        # DiffEqBase.solve(prob, _alg, args...; kwargs...)
        integrator = DiffEqBase.__init(prob, _alg, args...; kwargs...)
    end

    integrator.dt = stepsize(integrator.cache.uT, integrator.opts.abstol) # override handle_dt! setting of initial dt
    DiffEqBase.solve!(integrator)
    integrator.sol
end

# used in continuous callbacks and related methods to update Taylor expansions cache
function update_jetcoeffs_cache!(u,f,p,cache::TaylorMethodCache)
    @unpack tT, uT, duT, uauxT, parse_eqs = cache
    for i in eachindex(u)
        @inbounds uT[i][0] = u[i]
        duT[i].coeffs .= zero(duT[i][0])
    end
    __jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p)
    return nothing
end

# This function was modified from OrdinaryDiffEq.jl; MIT-licensed
# DiffEqBase.addsteps! overload for ::TaylorMethodCache to handle continuous
# and vector callbacks with TaylorIntegration.jl via the common interface
function DiffEqBase.addsteps!(k,t,uprev,u,dt,f,p,cache::TaylorMethodCache,always_calc_begin = false,allow_calc_end = true,force_calc_end = false)
    if length(k)<2 || always_calc_begin
        if typeof(cache) <: OrdinaryDiffEqMutableCache
        rtmp = similar(u, eltype(eltype(k)))
        f(rtmp,uprev,p,t)
        copyat_or_push!(k,1,rtmp)
        f(rtmp,u,p,t+dt)
        copyat_or_push!(k,2,rtmp)
        else
        copyat_or_push!(k,1,f(uprev,p,t))
        copyat_or_push!(k,2,f(u,p,t+dt))
        end
    end
    update_jetcoeffs_cache!(u,f,p,cache)
    nothing
end

# This function was modified from TaylorSeries.jl; MIT-licensed
# evaluate! overload to handle AbstractArray
import TaylorSeries: evaluate!
function evaluate!(x::AbstractArray{Taylor1{T},N}, δt::S,
        x0::AbstractArray{T,N}) where {T<:Number, S<:Number, N}

    # @assert length(x) == length(x0)
    @inbounds for i in eachindex(x, x0)
        x0[i] = evaluate( x[i], δt )
    end
    nothing
end
