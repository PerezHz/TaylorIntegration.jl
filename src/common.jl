using DiffEqBase, OrdinaryDiffEq
using StaticArrays: SVector, SizedArray
using RecursiveArrayTools: ArrayPartition

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
    parse_eqs::Bool
end

TaylorMethod(order; parse_eqs=true) = TaylorMethod(order, parse_eqs) # set `parse_eqs` to `true` by default

alg_order(alg::TaylorMethod) = alg.order

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

struct TaylorMethodConstantCache{uTType} <: OrdinaryDiffEqConstantCache
    uT::uTType
    parse_eqs::Ref{Bool}
end

function alg_cache(alg::TaylorMethod, u, rate_prototype, uEltypeNoUnits,
        uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p,
        calck,::Val{true})
    tT = Taylor1(typeof(t), alg.order)
    tT[0] = t
    uT = Taylor1.(u, tT.order)
    duT = zero.(Taylor1.(u, tT.order))
    uauxT = similar(uT)
    TaylorMethodCache(
        u,
        uprev,
        similar(u),
        zero(rate_prototype),
        zero(rate_prototype),
        tT,
        uT,
        duT,
        uauxT,
        Ref(alg.parse_eqs)
        )
end

alg_cache(alg::TaylorMethod, u, rate_prototype, uEltypeNoUnits,
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

stepsize_controller!(integrator,alg::TaylorMethod) = stepsize(integrator.cache.uT, integrator.opts.abstol)
step_accept_controller!(integrator, alg::TaylorMethod, q) = q

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

    f = prob.f
    if !isinplace && typeof(prob.u0) <: AbstractArray
        if prob.f isa DynamicalODEFunction
            f1! = (dv, v, u, p, t) -> (dv .= prob.f.f1(v, u, p, t); 0)
            f2! = (du, v, u, p, t) -> (du .= prob.f.f2(v, u, p, t); 0)
            _alg = TaylorMethod(alg.order, parse_eqs = false)
            ### workaround use of `SVector` with oop `DynamicalODEProblem`
            ### TODO: add proper support for oop problems with arrays
            if eltype(prob.u0.x) <: SVector
                _u0 = ArrayPartition(SizedArray{Tuple{length(prob.u0.x[1])}}.(prob.u0.x))
            else
                _u0 = prob.u0
            end
            _prob = DynamicalODEProblem(f1!, f2!, _u0.x[1], _u0.x[2], prob.tspan, prob.p; prob.kwargs...)
        else
            f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
            _alg = TaylorMethod(alg.order, parse_eqs = false)
            _prob = ODEProblem(f!, prob.u0, prob.tspan, prob.p; prob.kwargs...)
        end
    elseif haskey(kwargs, :parse_eqs)
        _alg = TaylorMethod(alg.order, parse_eqs = kwargs[:parse_eqs])
        _prob = prob
    else
        _alg = TaylorMethod(alg.order)
        _prob = prob
    end

    # DiffEqBase.solve(prob, _alg, args...; kwargs...)
    integrator = DiffEqBase.__init(_prob, _alg, args...; kwargs...)
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
function DiffEqBase.addsteps!(k, t, uprev, u, dt, f, p, cache::TaylorMethodCache,
        always_calc_begin = false, allow_calc_end = true,force_calc_end = false)
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

@inline __jetcoeffs!(::Val{false}, f::ODEFunction, t, x, params) =
    jetcoeffs!(f.f, t, x, params)
@inline __jetcoeffs!(::Val{true},  f::ODEFunction, t, x, params) =
    jetcoeffs!(Val(f.f), t, x, params)
@inline __jetcoeffs!(::Val{false}, f::ODEFunction, t, x, dx, xaux, params) =
    jetcoeffs!(f.f, t, x, dx, xaux, params)
@inline __jetcoeffs!(::Val{true},  f::ODEFunction, t, x, dx, xaux, params) =
    jetcoeffs!(Val(f.f), t, x, dx, params)

_determine_parsing!(parse_eqs::Bool, f::ODEFunction, t, x, params) = _determine_parsing!(parse_eqs::Bool, f.f, t, x, params)
_determine_parsing!(parse_eqs::Bool, f::ODEFunction, t, x, dx, params) = _determine_parsing!(parse_eqs::Bool, f.f, t, x, dx, params)
