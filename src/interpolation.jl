# This file is part of the TaylorIntegration.jl package; MIT licensed
@auto_hash_equals struct TaylorInterpolant{T,U,N}
    t::AbstractVector{T}
    x::AbstractArray{Taylor1{U},N}
    #Inner constructor
    function TaylorInterpolant{T,U,N}(
        t::AbstractVector{T},
        x::AbstractArray{Taylor1{U},N}
    ) where {T<:Real, U<:Number, N}
        @assert size(x)[1] == length(t)-1
        @assert issorted(t) || issorted(t, rev=true)
        return new{T,U,N}(t, x)
    end
end

#outer constructor
function TaylorInterpolant(t::AbstractVector{T},
        x::AbstractArray{Taylor1{U},N}) where {T<:Real, U<:Number, N}
    return TaylorInterpolant{T,U,N}(t, x)
end

# return time vector index corresponding to interpolation range
function getinterpindex(tinterp::TaylorInterpolant{T,U,N}, t::T) where {T<:Real, U<:Number, N}
    tmin, tmax = minmax(tinterp.t[end], tinterp.t[1])
    @assert tmin ≤ t ≤ tmax "Evaluation time outside range of interpolation"
    if issorted(tinterp.t) # forward integration
        ind = searchsortedlast(tinterp.t, t)
    elseif issorted(tinterp.t, rev=true) # backward integration
        ind = searchsortedlast(tinterp.t, t, rev=true)
    end
    return ind
end

# function-like (callability) methods
function (tinterp::TaylorInterpolant{T,U,1})(t::T) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, t)
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1](δt)
    elseif tinterp.t[ind] == t
        return tinterp.x[ind]()
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,1})(t::Taylor1{T}) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, constant_term(t))
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1](δt)
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,1})(t::TaylorN{T}) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, constant_term(t))
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1](δt)
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,2})(t::T) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, t)
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1,:](δt)
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind,:](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,2})(t::Taylor1{T}) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, constant_term(t))
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1,:](δt)
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind,:](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,2})(t::TaylorN{T}) where {T<:Real, U<:Number}
    ind = getinterpindex(tinterp, constant_term(t))
    if ind == lastindex(tinterp.t)
        δt = t-tinterp.t[ind-1]
        return tinterp.x[ind-1,:](δt)
    else
        δt = t-tinterp.t[ind]
        return tinterp.x[ind,:](δt)
    end
end

function (tinterp::TaylorInterpolant{T,U,N})(t::V) where {T<:Real, U<:Number, V<:Real, N}
    R = promote_type(T, V)
    return tinterp(convert(R, t))
end
