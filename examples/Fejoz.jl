using TaylorIntegration

const k = 16
const n = 20
const ζ =  set_variables("ζ", numvars=2, order=k+1)


function X(dz, z, p, t)
    dz[1] = -sin(z[2])
    dz[2] = z[1]
    return dz
end

# Initial condition
z0 = [0.5, 0.0]
dz = zero(z0)

X(dz, z0, nothing, 0.0)

# Integration
t, z, zt = taylorinteg(X, z0, 0.0, 10.0, n, 1.e-16, Val(true))


# Integration including the small quantities
ζ =  set_variables("ζ", numvars=2, order=k+1)
zζ0 = z0 .+ ζ

tζ, zζ, zζt = taylorinteg(X, zζ0, 0.0, 10.0, n, 1.e-16, Val(true))


#---
# ζ =  set_variables("ζ", numvars=2, order=k+1)
z0 = [0.5, 0.0]
dz0 = zero(z0)

# Pendulum
H(z) = z[1]^2/2 - cos(z[2])

G(z) = ∇(H(z .+ ζ))
# G(z) = ∇(H(z))

# X(z) = [-G(z)[2], G(z)[1]]
X(z) = [0 -1; 1 0]*G(z)

X(z0)

# ξt = TaylorN(zero(Taylor1(n)), k+1)

# tξ = Taylor1(zero(ζ[1]), n)

t = Taylor1(n)
zt = z0 .+ t

function X(dz, z, p, t)
    z0 = constant_term.(z)
    # Taylor expansion of the vector field around z0
    xx = X(z0)
    @show(xx)
    # Deviation from expansion point (Taylor1 expansion)!
    δz = z .- z0
    # Split this into the equations of motion
    dz[1] = evaluate(xx[1], δz )
    dz[2] = evaluate(xx[2], δz )
    # dz .= evaluate.(xx, Ref{}(δz))
    return dz
end

dzt = deepcopy(zt)
zaux = deepcopy(zt)

X(dz0, z0, nothing, t)

X(dzt, zt, nothing, t)

TaylorIntegration.jetcoeffs!(X, t, zt, dzt, zaux, nothing)

taylorinteg(X, z0, 0.0, 10.0, n, 1.e-20, Val(true));
