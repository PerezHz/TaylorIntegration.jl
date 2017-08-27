# This file is part of the TaylorIntegration.jl package; MIT licensed

# stepsize
function stepsize{T<:Real}(x::Taylor1{Taylor1{T}}, epsilon::T)
    ord = x.order
    h = convert(T, Inf)
    for k in (ord-1, ord)
        @inbounds aux = Array{T}(x[k+1].order)
        for i in 1:x[k+1].order
            @inbounds aux[i] = norm(x[k+1][i],Inf)
        end
        aux == zeros(T, length(aux)) && continue
        aux = epsilon ./ aux
        kinv = one(T)/k
        aux = aux.^kinv
        h = min(h, minimum(aux))
    end
    return h
end

function stepsize{T<:Real}(q::Array{Taylor1{Taylor1{T}},1}, epsilon::T)
    h = convert(T, Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end

function stepsize{T<:Real}(x::Taylor1{TaylorN{T}}, epsilon::T)
    ord = x.order
    h = convert(T, Inf)
    for k in (ord-1, ord)
        @inbounds aux = Array{T}(x[k+1].order)
        for i in 1:x[k+1].order
            @inbounds aux[i] = norm(x[k+1][i].coeffs,Inf)
        end
        aux == zeros(T, length(aux)) && continue
        aux = epsilon ./ aux
        kinv = one(T)/k
        aux = aux.^kinv
        h = min(h, minimum(aux))
    end
    return h
end

function stepsize{T<:Real}(q::Array{Taylor1{TaylorN{T}},1}, epsilon::T)
    h = convert(T, Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end
