# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    stabilitymatrix!(eqsdiff!, t0, x, δx, dδx, jac)

Updates the matrix `jac::Array{T,2}` (linearized equations of motion)
computed from the equations of motion (`eqsdiff!`), at time `t0`
at `x`; `x` is of type `Vector{T<:Number}`. `δx` and `dδx` are two
auxiliary arrays of type `Vector{TaylorN{T}}` to avoid allocations.

"""
function stabilitymatrix!(eqsdiff!, t::Taylor1{T},
        x::SubArray{Taylor1{U},1}, δx::Array{TaylorN{Taylor1{U}},1},
        dδx::Array{TaylorN{Taylor1{U}},1}, jac::Array{Taylor1{U},2}) where {T<:Real, U<:Number}
    for ind in eachindex(x)
        @inbounds δx[ind] = x[ind] + TaylorN(Taylor1{U},ind,order=1)
    end
    eqsdiff!(t, δx, dδx)
    jacobian!( jac, dδx )
    nothing
end

# Modified from `cgs` and `mgs`, obtained from:
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
# Classical Gram–Schmidt (Trefethen algorithm 7.1), implemented in the simplest way
# (We could make it faster by unrolling loops to avoid temporaries arrays etc.)
function classicalGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, aⱼ)
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end

# Modified Gram–Schmidt (Trefethen algorithm 8.1); see also
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
function modifiedGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, vⱼ) # ⟵ NOTICE: mgs has vⱼ, clgs has aⱼ
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end

"""
    lyap_jetcoeffs!(eqsdiff!, t, x, dx, xaux, δx, dδx, jac, _δv[, eqsdiff_jac!])

Similar to [`jetcoeffs!`](@ref) for the calculation of the Lyapunov spectrum.
`jac` is the current value of the linearization of the equations of motion
(i.e., the Jacobian), and `xaux`, `δx`, `dδx` and `_δv` are auxiliary vectors.
Optionally, the user may provide an Jacobian function `eqsdiff_jac!` to
evaluate in-place the Jacobian; the user-defined Jacobian function must have the
call signature `eqsdiff_jac!(jac, t, x, dx) -> nothing`. Otherwise, the current
value of the Jacobian is computed via automatic differentiation using
`TaylorSeries.jl`.

"""
function lyap_jetcoeffs!(eqsdiff!, t::Taylor1{T}, x::Vector{Taylor1{U}},
        dx::Vector{Taylor1{U}}, xaux::Vector{Taylor1{U}},
        δx::Array{TaylorN{Taylor1{U}},1}, dδx::Array{TaylorN{Taylor1{U}},1},
        jac::Array{Taylor1{U},2}, _δv::Array{TaylorN{Taylor1{U}}},
        eqsdiff_jac! =Nothing) where {T<:Real, U<:Number}
    order = x[1].order
    # Dimensions of phase-space: dof
    nx = length(x)
    dof = length(δx)
    for ord in 0:order-1
        ordnext = ord+1
        # Set `taux`, auxiliary Taylor1 variable to order `ord`
        @inbounds taux = Taylor1( t.coeffs[1:ordnext] )
        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ordnext] )
        end

        if eqsdiff_jac! == Nothing
            # Set δx equal to current value of xaux plus 1st-order variations
            for ind in eachindex(δx)
                @inbounds δx[ind] = xaux[ind] + _δv[ind]
            end
            # Equations of motion
            # TODO! define a macro to optimize the eqsdiff
            eqsdiff!(taux, δx, dδx)
            @inbounds dx[1:dof] .= constant_term.(dδx)
            # Stability matrix
            jacobian!(jac, dδx)
        else
            # Equations of motion
            # TODO! define a macro to optimize the eqsdiff
            eqsdiff!(taux, xaux, dx)
            # Stability matrix
            eqsdiff_jac!(jac, taux, xaux, dx)
        end

        @inbounds dx[dof+1:nx] = jac * reshape( xaux[dof+1:nx], (dof,dof) )
        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j][ordnext] = dx[j][ord]/ordnext
        end
    end
    nothing
end

"""
    lyap_taylorstep!(f!, t, x, dx, xaux, δx, dδx, jac, t0, t1, x0, order, abstol, _δv[, f])

Similar to [`taylorstep!`](@ref) for the calculation of the Lyapunov spectrum.
`jac` is the current value of the linearization of the equations of motion, i.e,
the Jacobian. `xaux`, `δx`, `dδx` and `vT` are auxiliary vectors. Optionally, the
user may provide an Jacobian function `f!` to evaluate the current value of the
Jacobian. For more details on `f!`, see [`lyap_jetcoeffs!`](@ref).

"""
function lyap_taylorstep!(f!, t::Taylor1{T}, x::Vector{Taylor1{U}},
        dx::Vector{Taylor1{U}}, xaux::Vector{Taylor1{U}},
        δx::Array{TaylorN{Taylor1{U}},1}, dδx::Array{TaylorN{Taylor1{U}},1},
        jac::Array{Taylor1{U},2}, t0::T, t1::T, x0::Array{U,1}, order::Int,
        abstol::T, _δv::Array{TaylorN{Taylor1{U}}}, jac! =Nothing) where {T<:Real, U<:Number}

    lyap_jetcoeffs!(f!, t, x, dx, xaux, δx, dδx, jac, _δv, jac!)

    # Dimensions of phase-space: dof
    dof = length(δx)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(view(x, 1:dof), abstol)
    δt = min(δt, t1-t0)

    # Update x0
    evaluate!(x, δt, x0)
    return δt
end

"""
    lyap_taylorinteg(f!, q0, t0, tmax, order, abstol[, f!]; maxsteps::Int=500)

Similar to [`taylorinteg!`](@ref) for the calculation of the Lyapunov
spectrum. Note that the number of `TaylorN` variables should be set
previously by the user (e.g., by means of `TaylorSeries.set_variables`) and
should be equal to the length of the vector of initial conditions `q0`.
Otherwise, whenever `length(q0) != TaylorSeries.get_numvars()`, then
`lyap_taylorinteg` throws an `AssertionError`.  Optionally, the user may provide
an Jacobian function `f!` to evaluate the current value of the Jacobian.
Otherwise, the current value of the Jacobian is computed via automatic
differentiation using `TaylorSeries.jl`. For more details on `f!`, see
[`lyap_jetcoeffs!`](@ref).

"""
function lyap_taylorinteg(f!, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T, jac! =Nothing; maxsteps::Int=500) where {T<:Real, U<:Number}
    # Allocation
    tv = Array{T}(undef, maxsteps+1)
    dof = length(q0)
    xv = Array{U}(undef, dof, maxsteps+1)
    λ = similar(xv)
    λtsum = similar(q0)
    jt = Matrix{U}(I, dof, dof)
    _δv = Array{TaylorN{Taylor1{U}}}(undef, dof)

    # Check only if user does not provide Jacobian
    if jac! == Nothing
        @assert get_numvars() == dof "`length(q0)` must be equal to number of variables set by `TaylorN`"
    end

    for ind in eachindex(q0)
        _δv[ind] = TaylorN(Taylor1{U},ind,order=1)
    end

    # Initial conditions
    @inbounds tv[1] = t0
    for ind in eachindex(q0)
        @inbounds xv[ind,1] = q0[ind]
        @inbounds λ[ind,1] = zero(U)
        @inbounds λtsum[ind] = zero(U)
    end
    x0 = vcat(q0, reshape(jt, dof*dof))
    nx0 = dof*(dof+1)
    t00 = t0

    # Initialize the vector of Taylor1 expansions
    t = Taylor1(T, order)
    x = Array{Taylor1{U}}(undef, nx0)
    @inbounds x .= Taylor1.( x0, order )
    @inbounds t[0] = t0

    #Allocate auxiliary arrays
    dx = Array{Taylor1{U}}(undef, nx0)
    xaux = Array{Taylor1{U}}(undef, nx0)
    δx = Array{TaylorN{Taylor1{U}}}(undef, dof)
    dδx = Array{TaylorN{Taylor1{U}}}(undef, dof)
    jac = Array{Taylor1{U}}(undef, dof,dof)
    fill!(jac, zero(x[1]))
    QH = Array{U}(undef, dof,dof)
    RH = Array{U}(undef, dof,dof)
    aⱼ = Array{U}(undef, dof )
    qᵢ = similar(aⱼ)
    vⱼ = similar(aⱼ)

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = lyap_taylorstep!(f!, t, x, dx, xaux, δx, dδx, jac, t0, tmax, x0, order, abstol, _δv, jac!)
        for ind in eachindex(jt)
            @inbounds jt[ind] = x0[dof+ind]
        end
        modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
        t0 += δt
        @inbounds t[0] = t0
        tspan = t0-t00
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds for ind in eachindex(q0)
            xv[ind,nsteps] = x0[ind]
            λtsum[ind] += log(RH[ind,ind])
            λ[ind,nsteps] = λtsum[ind]/tspan
        end
        for ind in eachindex(QH)
            @inbounds x0[dof+ind] = QH[ind]
        end
        @inbounds x .= Taylor1.( x0, order )
        if nsteps > maxsteps
            @info("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps),  view(transpose(xv),1:nsteps,:),  view(transpose(λ),1:nsteps,:)
end

function lyap_taylorinteg(f!, q0::Array{U,1}, trange::Union{AbstractRange{T},Vector{T}},
        order::Int, abstol::T, jac! =Nothing; maxsteps::Int=500) where {T<:Real, U<:Number}
    # Allocation
    nn = length(trange)
    dof = length(q0)
    xv = Array{U}(undef, dof, nn)
    fill!(xv, U(NaN))
    λ = similar(xv)
    λtsum = similar(q0)
    jt = Matrix{U}(I, dof, dof)
    _δv = Array{TaylorN{Taylor1{U}}}(undef, dof)

    # Check only if user does not provide Jacobian
    if jac! == Nothing
        @assert get_numvars() == dof "`length(q0)` must be equal to number of variables set by `TaylorN`"
    end

    for ind in eachindex(q0)
        _δv[ind] = TaylorN(Taylor1{U},ind,order=1)
    end

    # Initial conditions
    @inbounds for ind in eachindex(q0)
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(U)
        λtsum[ind] = zero(U)
    end

    # Initialize the vector of Taylor1 expansions
    t = Taylor1(T, order)
    x0 = vcat(q0, reshape(jt, dof*dof))
    nx0 = length(x0)
    x = Array{Taylor1{U}}(undef, nx0)
    @inbounds x .= Taylor1.( x0, order )
    @inbounds t[0] = trange[1]
    t00 = trange[1]
    tspan = zero(T)

    #Allocate auxiliary arrays
    dx = Array{Taylor1{U}}(undef, nx0)
    xaux = Array{Taylor1{U}}(undef, nx0)
    δx = Array{TaylorN{Taylor1{U}}}(undef, dof)
    dδx = Array{TaylorN{Taylor1{U}}}(undef, dof)
    jac = Array{Taylor1{U}}(undef, dof,dof)
    fill!(jac, zero(x[1]))
    QH = Array{U}(undef, dof,dof)
    RH = Array{U}(undef, dof,dof)
    aⱼ = Array{U}(undef, dof )
    qᵢ = similar(aⱼ)
    vⱼ = similar(aⱼ)

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt = lyap_taylorstep!(f!, t, x, dx, xaux, δx, dδx, jac, t0, t1, x0, order, abstol, _δv, jac!)
            for ind in eachindex(jt)
                @inbounds jt[ind] = x0[dof+ind]
            end
            modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
            t0 += δt
            @inbounds t[0] = t0
            nsteps += 1
            @inbounds for ind in eachindex(q0)
                λtsum[ind] += log(RH[ind,ind])
            end
            for ind in eachindex(QH)
                @inbounds x0[dof+ind] = QH[ind]
            end
            @inbounds x .= Taylor1.( x0, order )
            t0 ≥ t1 && break
        end
        if nsteps ≥ maxsteps && t0 != t1
            @info("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        tspan = t0-t00
        @inbounds for ind in eachindex(q0)
            xv[ind,iter] = x0[ind]
            λ[ind,iter] = λtsum[ind]/tspan
        end
    end

    return transpose(xv),  transpose(λ)
end
