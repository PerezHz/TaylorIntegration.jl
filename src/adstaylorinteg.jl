using TaylorSeries: generate_index_vectors
using LsqFit: curve_fit
using AbstractTrees: Leaves, getroot

# In-place exponential model y(t) = A * exp(B * t),
# used within size_per_variable to estimate the error
# of each jet transport variable
exp_model!(F, t, p) = (@. F = p[1] * exp(p[2] * t))

# In place jacobian of exp_model! wrt parameters p
function exp_model_jacobian!(J::Array{T, 2}, t, p) where {T <: Real}
    @. J[:, 1] = exp(p[2] * t)
    @. @views J[:, 2] = t * p[1] * J[:, 1]
end

# Estimate the error of each jet transport variable
@doc raw"""
    size_per_variable(P::TaylorN{T}) where {T <: Real}
    size_per_variable(P::AbstractVector{TaylorN{T}}) where {T <: Real}

Return a `Vector{T}` with the error estimation of each jet transport variable in `P`.

!!! reference
    See section 3 of https://doi.org/10.1007/s10569-015-9618-3.
"""
function size_per_variable(P::TaylorN{T}) where {T <: Real}
    # Jet transport order
    varorder = P.order
    # Number of variables
    nv = get_numvars()
    # Absolute sum per variable per order
    ys = zeros(T, nv, varorder+1)
    
    for i in eachindex(P.coeffs)
        idxs = generate_index_vectors(nv, i-1)
        for j in eachindex(idxs)
            for k in eachindex(idxs[j])
                ys[k, idxs[j][k]+1] += abs(P.coeffs[i].coeffs[j])
            end
        end
    end
    # Initial parameters
    p0 = ones(T, 2)
    # Orders
    xs = collect(0:varorder)
    
    M = Vector{T}(undef, nv)
    for i in eachindex(M)
        # Non zero coefficients
        idxs = findall(!iszero, view(ys, i, :))
        # Fit exponential model
        fit = curve_fit(exp_model!, exp_model_jacobian!, view(xs, idxs),
                        view(ys, i, idxs), p0; inplace = true)
        # Estimate next order coefficient
        exp_model!(view(M, i:i), varorder+1, fit.param)
    end

    return M
end

function size_per_variable(P::AbstractVector{TaylorN{T}}) where {T <: Real}
    # Number of variables
    nv = get_numvars()
    # Size per variable per element of P
    norms = Matrix{T}(undef, nv, length(P))
    for i in eachindex(P)
        norms[:, i] .= size_per_variable(P[i])
    end
    # Sum over element of P
    M = sum(norms, dims = 2)

    return M
end

@doc raw"""
    splitdirection(P::TaylorN{T}) where {T <: Real}
    splitdirection(P::AbstractVector{TaylorN{T}}) where {T <: Real}

Return the index of the jet transport variable with the largest error estimation
according to [`size_per_variable`](@ref).
"""
function splitdirection(P::TaylorN{T}) where {T <: Real}
    # Size per variable
    M = size_per_variable(P)
    # Variable with maximum error
    return argmax(M)[1]
end

function splitdirection(P::AbstractVector{TaylorN{T}}) where {T <: Real}
    # Size per variable
    M = size_per_variable(P)
    # Variable with maximum error
    return argmax(M)[1]
end

# Split node's domain in half
# See section 3 of https://doi.org/10.1007/s10569-015-9618-3
function split(node::ADSBinaryNode{N, M, T}, x::SVector{M, TaylorN{T}},
               p::SVector{M, Taylor1{TaylorN{T}}}) where {N, M, T <: Real} 
    # Split direction
    j = splitdirection(x)
    # Split domain
    s1, s2 = split(node.s, j)
    # Jet transport variables
    v_1, v_2 = get_variables(), get_variables()
    # Shift expansion point
    r = getroot(node).s
    v_1[j] = v_1[j]/2 + r.lo[j]/2
    v_2[j] = v_2[j]/2 + r.hi[j]/2
    # Taylor1 order
    order = p[1].order
    # Left half
    x1 = SVector{M, TaylorN{T}}(x[i](v_1) for i in eachindex(x))
    p1 = SVector{M, Taylor1{TaylorN{T}}}(
        Taylor1( map(z -> z(v_1), p[i].coeffs), order ) for i in eachindex(p)
    )
    # Right half
    x2 = SVector{M, TaylorN{T}}(x[i](v_2) for i in eachindex(x))
    p2 = SVector{M, Taylor1{TaylorN{T}}}(
        Taylor1( map(z -> z(v_2), p[i].coeffs), order ) for i in eachindex(p)
    )

    return s1, x1, p1, s2, x2, p2
end

@doc raw"""
    split!(node::ADSBinaryNode{N, M, T}, p::SVector{M, Taylor1{TaylorN{T}}},
           dt::T, nsplits::Int, maxsplits::Int, stol::T) where {N, M, T <: Real}

Split `node` in half if at least one element of `p(dt)` has any of its last two
coefficients greater than `stol`.
"""
function split!(node::ADSBinaryNode{N, M, T}, p::SVector{M, Taylor1{TaylorN{T}}},
                dt::T, nsplits::Int, maxsplits::Int, stol::T) where {N, M, T <: Real}
    # Evaluate x at dt
    x = _eval(p, dt)
    # Split criteria for each element of x
    mask = map(y -> norm(y.coeffs[end-1:end], Inf), x)
    # Split
    if nsplits < maxsplits && any(mask .> stol)
        # Split
        s1, x1, p1, s2, x2, p2 = split(node, x, p)
        # Left half
        leftchild!(node, s1, node.t + dt, x1, p1)
        # Right half
        rightchild!(node, s2, node.t + dt, x2, p2)
        # Update number of splits
        nsplits += 1
    # No split
    else
        leftchild!(node, node.s, node.t + dt, x, p)
    end

    return nsplits
end

function taylorinteg(
    f!, q0::SVector{M, TaylorN{T}}, s::ADSDomain{N, T}, t0::T, tmax::T,
    order::Int, stol::T, abstol::T, params = nothing; maxsplits::Int = 10,
    maxsteps::Int = 500, parse_eqs::Bool = true
    ) where {N, M, T <: Real}

    # Initialize the vector of Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Array{Taylor1{TaylorN{T}}}(undef, M)
    dx = Array{Taylor1{TaylorN{T}}}(undef, M)
    @inbounds for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
        @inbounds dx[i] = Taylor1( zero(q0[i]), order )
    end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    if parse_eqs
        return _taylorinteg!(f!, q0, s, t0, tmax, order, stol, abstol, rv, 
                             params; maxsplits, maxsteps)
    else
        return _taylorinteg!(f!, q0, s, t0, tmax, order, stol, abstol,
                             params; maxsplits, maxsteps)
    end
end

function _taylorinteg!(
    f!, q0::SVector{M, TaylorN{T}}, s::ADSDomain{N, T}, t0::T, tmax::T,
    order::Int, stol::T, abstol::T, params = nothing; maxsplits::Int = 10,
    maxsteps::Int = 500
    ) where {N, M, T <: Real}

    # Allocation
    t = t0 + Taylor1( T, order )
    dt = zeros(T, maxsplits)
    x = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    dx = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    @inbounds for j in 1:maxsplits
        @inbounds for i in eachindex(q0)
            @inbounds x[i, j] = Taylor1( q0[i], order )
            @inbounds dx[i, j] = Taylor1( zero(q0[i]), order )
        end
    end
    nv = ADSBinaryNode{N, M, T}(s, t0, SVector{M, TaylorN{T}}(q0),
                                SVector{M, Taylor1{TaylorN{T}}}(x[:, 1]))
    
    nsteps = 1
    nsplits = 1
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    while sign_tstep*t0 < sign_tstep*tmax

        for (k, node) in enumerate(Leaves(nv))
            @inbounds for i in 1:M
                @inbounds x[i, k] = Taylor1( node.x[i], order )
                @inbounds dx[i, k] = Taylor1( zero(node.x[i]), order )
            end
            xaux = Vector{Taylor1{TaylorN{T}}}(undef, M)
            dt[k] = taylorstep!(f!, t, x[:, k], dx[:, k], xaux, abstol, params) # δt is positive!
            # Below, δt has the proper sign according to the direction of the integration
            dt[k] = sign_tstep * min(dt[k], sign_tstep*(tmax-t0))
        end

        δt = minimum(view(dt, 1:nsplits))
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1

        for (k, node) in enumerate(Leaves(nv))
            nsplits = split!(node, SVector{M, Taylor1{TaylorN{T}}}(x[:, k]),
                             δt, nsplits, maxsplits, stol)
        end

        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return nv
end 

function _taylorinteg!(
    f!, q0::SVector{M, TaylorN{T}}, s::ADSDomain{N, T}, t0::T, tmax::T,
    order::Int, stol::T, abstol::T, rv::RetAlloc{Taylor1{TaylorN{T}}},
    params = nothing; maxsplits::Int = 10, maxsteps::Int = 500
    ) where {N, M, T <: Real}

    # Allocation
    t = t0 + Taylor1( T, order )
    dt = zeros(T, maxsplits)
    x = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    dx = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    @inbounds for j in 1:maxsplits
        @inbounds for i in eachindex(q0)
            @inbounds x[i, j] = Taylor1( q0[i], order )
            @inbounds dx[i, j] = Taylor1( zero(q0[i]), order )
        end
    end
    nv = ADSBinaryNode{N, M, T}(s, t0, SVector{M, TaylorN{T}}(q0),
                                SVector{M, Taylor1{TaylorN{T}}}(x[:, 1]))

    nsteps = 1
    nsplits = 1
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    while sign_tstep*t0 < sign_tstep*tmax

        for (k, node) in enumerate(Leaves(nv))
            @inbounds for i in 1:M
                @inbounds x[i, k] = Taylor1( node.x[i], order )
                @inbounds dx[i, k] = Taylor1( zero(node.x[i]), order )
            end
            dt[k] = taylorstep!(f!, t, x[:, k], dx[:, k], abstol, params, rv) # δt is positive!
            # Below, δt has the proper sign according to the direction of the integration
            dt[k] = sign_tstep * min(dt[k], sign_tstep*(tmax-t0))
        end

        δt = minimum(view(dt, 1:nsplits))
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1

        for (k, node) in enumerate(Leaves(nv))
            nsplits = split!(node, SVector{M, Taylor1{TaylorN{T}}}(x[:, k]),
                             δt, nsplits, maxsplits, stol)
        end

        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return nv
end