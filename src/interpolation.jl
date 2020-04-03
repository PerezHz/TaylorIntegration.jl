# This file is part of the TaylorIntegration.jl package; MIT licensed
@auto_hash_equals struct TaylorInterpolant{T,U,N}
    t0::T
    t::AbstractVector{T}
    x::AbstractArray{Taylor1{U},N}
    #Inner constructor
    function TaylorInterpolant{T,U,N}(
        t0::T,
        t::AbstractVector{T},
        x::AbstractArray{Taylor1{U},N}
    ) where {T<:Real, U<:Number, N}
        @assert size(x)[1] == length(t)-1
        @assert issorted(t) || issorted(t, rev=true)
        return new{T,U,N}(t0, t, x)
    end
end

#outer constructor
function TaylorInterpolant(t0::T, t::AbstractVector{T},
        x::AbstractArray{Taylor1{U},N}) where {T<:Real, U<:Number, N}
    return TaylorInterpolant{T,U,N}(t0, t, x)
end

# function TaylorInterpolant(t::AbstractVector{T},
#         x::AbstractArray{Taylor1{U},N}) where {T<:Real, U<:Number, N}
#     return TaylorInterpolant{T,U,N}(t[1], t.-t[1], x)
# end

# return time vector index corresponding to interpolation range
function getinterpindex(tinterp::TaylorInterpolant{T,U,N}, t::T) where {T<:Real, U<:Number, N}
    tmin, tmax = minmax(tinterp.t[end], tinterp.t[1])
    Δt = t-tinterp.t0
    @assert tmin ≤ Δt ≤ tmax "Evaluation time outside range of interpolation"
    if Δt == tinterp.t[end] # compute solution at final time from last step expansion
        ind = lastindex(tinterp.t)-1
    elseif issorted(tinterp.t) # forward integration
        ind = searchsortedlast(tinterp.t, Δt)
    elseif issorted(tinterp.t, rev=true) # backward integration
        ind = searchsortedlast(tinterp.t, Δt, rev=true)
    end
    return ind, Δt
end

# function-like (callability) methods
function (tinterp::TaylorInterpolant{T,U,1})(t::T) where {T<:Real, U<:Number}
    ind, Δt = getinterpindex(tinterp, t)
    δt = Δt-tinterp.t[ind]
    return tinterp.x[ind](δt)
end

function (tinterp::TaylorInterpolant{T,U,1})(t::Taylor1{T}) where {T<:Real, U<:Number}
    ind, _ = getinterpindex(tinterp, constant_term(t))
    δt = (t-tinterp.t0)-tinterp.t[ind]
    return tinterp.x[ind](δt)
end

function (tinterp::TaylorInterpolant{T,U,1})(t::TaylorN{T}) where {T<:Real, U<:Number}
    ind, _ = getinterpindex(tinterp, constant_term(t))
    δt = (t-tinterp.t0)-tinterp.t[ind]
    return tinterp.x[ind](δt)
end

function (tinterp::TaylorInterpolant{T,U,2})(t::T) where {T<:Real, U<:Number}
    ind, Δt = getinterpindex(tinterp, t)
    δt = Δt-tinterp.t[ind]
    return tinterp.x[ind,:](δt)
end

function (tinterp::TaylorInterpolant{T,U,2})(t::Taylor1{T}) where {T<:Real, U<:Number}
    ind, _ = getinterpindex(tinterp, constant_term(t))
    δt = (t-tinterp.t0)-tinterp.t[ind]
    return tinterp.x[ind,:](δt)
end

function (tinterp::TaylorInterpolant{T,U,2})(t::TaylorN{T}) where {T<:Real, U<:Number}
    ind, _ = getinterpindex(tinterp, constant_term(t))
    δt = (t-tinterp.t0)-tinterp.t[ind]
    return tinterp.x[ind,:](δt)
end

function (tinterp::TaylorInterpolant{T,U,N})(t::V) where {T<:Real, U<:Number, V<:Real, N}
    R = promote_type(T, V)
    return tinterp(convert(R, t))
end
