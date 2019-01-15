# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: norm

# Constants for the integrations
const _order = 20
const _abstol = 1.0E-20
const t0 = 0.0
const tf = 1000.0

# Scalar integration
global ll = 2

@testset "Scalar case: xdot(t,x) = b-x^2" begin
    b1 = 3.0
    @taylorize xdot1(t, x) = b1-x^2
    @test length(methods(TaylorIntegration.jetcoeffs!)) == ll+1
    @test (@isdefined xdot1)

    x0 = 1.0
    tv1, xv1 = taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    @time taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv1p, xv1p = taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000)
    @time taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )
end
@testset "Scalar case: xdot(t,x) = b-x^2 with `local` constant" begin
    @taylorize xdot11(t, x) = (local b1 = 3; b1-x^2)
    # @test length(methods(TaylorIntegration.jetcoeffs!)) == ll+1
    @test (@isdefined xdot11)

    x0 = 1.0
    tv1, xv1 = taylorinteg( xdot11, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    @time taylorinteg( xdot11, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv1p, xv1p = taylorinteg( xdot11, x0, t0, tf, _order, _abstol, maxsteps=1000)
    @time taylorinteg( xdot11, x0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )
end


@testset "Scalar case: xdot(t,x) = -9.81" begin
    xdot2(t, x) = -9.81 + zero(t)
    @taylorize xdot2_parsed(t, x) = -9.81 #(local ggrav = -9.81; ggrav)

    @test (@isdefined xdot2_parsed)

    tv11, xv11   = taylorinteg( xdot2, 10.0, 1.0, tf, _order, _abstol)
    @time taylorinteg( xdot2, 10.0, 1.0, tf, _order, _abstol)
    tv11p, xv11p = taylorinteg( xdot2_parsed, 10.0, 1.0, tf, _order, _abstol)
    @time taylorinteg( xdot2_parsed, 10.0, 1.0, tf, _order, _abstol)

    @test length(tv11) == length(tv11p)
    @test iszero( norm(tv11-tv11p, Inf) )
    @test iszero( norm(xv11-xv11p, Inf) )
end
@testset "Scalar case: xdot(t,x) = -9.81 with `local` constant" begin
    xdot22(t, x) = -9.81 + zero(t)
    @taylorize xdot22_parsed(t, x) = (local ggrav = -9.81; ggrav + zero(t))

    @test (@isdefined xdot22_parsed)

    tv11, xv11   = taylorinteg( xdot22, 10.0, 1.0, tf, _order, _abstol)
    @time taylorinteg( xdot22, 10.0, 1.0, tf, _order, _abstol)
    tv11p, xv11p = taylorinteg( xdot22_parsed, 10.0, 1.0, tf, _order, _abstol)
    @time taylorinteg( xdot22_parsed, 10.0, 1.0, tf, _order, _abstol)

    @test length(tv11) == length(tv11p)
    @test iszero( norm(tv11-tv11p, Inf) )
    @test iszero( norm(xv11-xv11p, Inf) )
end


# Pendulum integrtf = 100.0
@testset "Integration of the pendulum" begin
    @taylorize function pendulum!(t, x, dx)
        dx[1] = x[2]
        dx[2] = -sin( x[1] )
        nothing
    end

    @test (@isdefined pendulum!)
    q0 = [pi-0.001, 0.0]
    tv2, xv2 = taylorinteg(pendulum!, q0, t0, tf, _order, _abstol, parse_eqs=false)
    @time taylorinteg(pendulum!, q0, t0, tf, _order, _abstol, parse_eqs=false)
    tv2p, xv2p = taylorinteg(pendulum!, q0, t0, tf, _order, _abstol)
    @time taylorinteg(pendulum!, q0, t0, tf, _order, _abstol)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )
end


# Complex dependent variables
@testset "Complex dependent variable" begin
    cc = complex(0.0,1.0)
    @taylorize eqscmplx(t,x) = cc*x

    @test (@isdefined eqscmplx)
    cx0 = complex(1.0, 0.0)
    tv3, xv3 = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    @time taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    tv3p, xv3p = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500)
    @time taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500)

    @test length(tv3) == length(tv3p)
    @test iszero( norm(tv3-tv3p, Inf) )
    @test iszero( norm(xv3-xv3p, Inf) )
end
@testset "Complex dependent variable with `local` constant" begin
    # cc = complex(0.0,1.0)
    @taylorize eqscmplx(t,x) = (local cc = Complex(0.0,1.0); cc*x)

    @test (@isdefined eqscmplx)
    cx0 = complex(1.0, 0.0)
    tv3, xv3 = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    @time taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    tv3p, xv3p = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500)
    @time taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500)

    @test length(tv3) == length(tv3p)
    @test iszero( norm(tv3-tv3p, Inf) )
    @test iszero( norm(xv3-xv3p, Inf) )
end


@testset "Time-dependent integration (with and without `local` vars)" begin
    @taylorize function integ_cos1(t, x)
        y = cos(t)
        return y
    end
    @taylorize function integ_cos2(t, x)
        local y = cos(t)
        yy = y
        return yy
    end

    @test (@isdefined integ_cos1)
    @test (@isdefined integ_cos2)

    tv11, xv11 = taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    @time taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    tv12, xv12 = taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol)
    @time taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol)
    @test length(tv11) == length(tv12)
    @test iszero( norm(tv11-tv12, Inf) )
    @test iszero( norm(xv11-xv12, Inf) )

    tv21, xv21 = taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    @time taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    tv22, xv22 = taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol)
    @time taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol)
    @test length(tv21) == length(tv22)
    @test iszero( norm(tv21-tv22, Inf) )
    @test iszero( norm(xv21-xv22, Inf) )

    @test iszero( norm(xv12-xv22, Inf) )
end

# Multiple (3) pendula
@testset "Multiple pendula" begin
    NN = 3
    nnrange = 1:3
    @taylorize function multpendula!(t, x, dx)
        # NN = 3
        for i in nnrange
            dx[i] = x[NN+i]
            dx[i+NN] = -sin( x[i] )
        end
        return nothing
    end

    @test (@isdefined multpendula!)
    q0 = [pi-0.001, 0.0, pi-0.001, 0.0,  pi-0.001, 0.0]
    tv4, xv4 = taylorinteg(multpendula!, q0, t0, tf, _order, _abstol,
        maxsteps=1000, parse_eqs=false)
    @time taylorinteg(multpendula!, q0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv4p, xv4p = taylorinteg(multpendula!, q0, t0, tf, _order, _abstol,
        maxsteps=1000)
    @time taylorinteg(multpendula!, q0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv4) == length(tv4p)
    @test iszero( norm(tv4-tv4p, Inf) )
    @test iszero( norm(xv4-xv4p, Inf) )
end
@testset "Multiple pendula with `local` constant" begin
    # NN = 3
    # nnrange = 1:3
    @taylorize function multpendula!(t, x, dx)
        local NN = 3
        local nnrange = 1:NN
        for i in nnrange
            dx[i] = x[NN+i]
            dx[i+NN] = -sin( x[i] )
        end
        return nothing
    end

    @test (@isdefined multpendula!)
    q0 = [pi-0.001, 0.0, pi-0.001, 0.0,  pi-0.001, 0.0]
    tv4, xv4 = taylorinteg(multpendula!, q0, t0, tf, _order, _abstol,
        maxsteps=1000, parse_eqs=false)
    @time taylorinteg(multpendula!, q0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv4p, xv4p = taylorinteg(multpendula!, q0, t0, tf, _order, _abstol,
        maxsteps=1000)
    @time taylorinteg(multpendula!, q0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv4) == length(tv4p)
    @test iszero( norm(tv4-tv4p, Inf) )
    @test iszero( norm(xv4-xv4p, Inf) )
end


# Kepler problem
_order = 28
tf = 2π*100.0
@testset "Kepler problem (using `^`)" begin
    mμ = -1.0
    @taylorize function kepler1!(t, q, dq)
        r_p3d2 = (q[1]^2+q[2]^2)^1.5

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    NN = 2
    @taylorize function kepler1_parsed2!(t, q, dq)
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r_p3d2 = r2^(3/2)
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler1!)
    @test (@isdefined kepler1_parsed2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv5, xv5 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv5p, xv5p = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv6p, xv6p = taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv7p, xv7p = taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv5) == length(tv5p) == length(tv6p)
    @test iszero( norm(tv5-tv5p, Inf) )
    @test iszero( norm(xv5-xv5p, Inf) )
    @test iszero( norm(tv5-tv6p, Inf) )
    @test iszero( norm(xv5-xv6p, Inf) )
    @test iszero( norm(tv7p-tv6p, Inf) )
    @test iszero( norm(xv7p-xv6p, Inf) )
end
@testset "Kepler problem (using `^`) with `local` constant" begin
    @taylorize function kepler1!(t, q, dq)
        local mμ = -1.0
        r_p3d2 = (q[1]^2+q[2]^2)^1.5

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    @taylorize function kepler1_parsed2!(t, q, dq)
        local mμ = -1.0
        local NN = 2
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r_p3d2 = r2^(3/2)
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler1!)
    @test (@isdefined kepler1_parsed2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv5, xv5 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv5p, xv5p = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv6p, xv6p = taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv7p, xv7p = taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler1_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv5) == length(tv5p) == length(tv6p)
    @test iszero( norm(tv5-tv5p, Inf) )
    @test iszero( norm(xv5-xv5p, Inf) )
    @test iszero( norm(tv5-tv6p, Inf) )
    @test iszero( norm(xv5-xv6p, Inf) )
    @test iszero( norm(tv7p-tv6p, Inf) )
    @test iszero( norm(xv7p-xv6p, Inf) )
end


@testset "Kepler problem (using `sqrt`)" begin
    mμ = -1.0
    @taylorize function kepler2!(t, q, dq)
        r = sqrt(q[1]^2+q[2]^2)
        r_p3d2 = r^3

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end
    NN = 2
    @taylorize function kepler2_parsed2!(t, q, dq)
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r = sqrt(r2)
        r_p3d2 = r^3
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    # @test (@isdefined kepler2!)
    @test (@isdefined kepler2_parsed2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv5, xv5 = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv5p, xv5p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv6p, xv6p = taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv7p, xv7p = taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv5) == length(tv5p) == length(tv6p)
    @test iszero( norm(tv5-tv5p, Inf) )
    @test iszero( norm(xv5-xv5p, Inf) )
    @test iszero( norm(tv5-tv6p, Inf) )
    @test iszero( norm(xv5-xv6p, Inf) )
    @test iszero( norm(tv7p-tv6p, Inf) )
    @test iszero( norm(xv7p-xv6p, Inf) )
end
@testset "Kepler problem (using `sqrt`) with `local` constants" begin
    @taylorize function kepler2!(t, q, dq)
        local mμ = -1.0
        r = sqrt(q[1]^2+q[2]^2)
        r_p3d2 = r^3

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    @taylorize function kepler2_parsed2!(t, q, dq)
        local NN = 2
        local mμ = -1.0
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r = sqrt(r2)
        r_p3d2 = r^3
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    # @test (@isdefined kepler2!)
    @test (@isdefined kepler2_parsed2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv5, xv5 = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv5p, xv5p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv6p, xv6p = taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    @time taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv7p, xv7p = taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)
    @time taylorinteg(kepler2_parsed2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv5) == length(tv5p) == length(tv6p)
    @test iszero( norm(tv5-tv5p, Inf) )
    @test iszero( norm(xv5-xv5p, Inf) )
    @test iszero( norm(tv5-tv6p, Inf) )
    @test iszero( norm(xv5-xv6p, Inf) )
    @test iszero( norm(tv7p-tv6p, Inf) )
    @test iszero( norm(xv7p-xv6p, Inf) )
end


tf = 20.0
@testset "Lyapunov spectrum and `@taylorize`" begin
    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Lorenz system ODE:
    @taylorize function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end
    # @show(methods(TaylorIntegration.jetcoeffs!))

    #Lorenz system Jacobian (in-place):
    function lorenz_jac!(jac, t, x)
        jac[1,1] = -σ+zero(x[1])
        jac[2,1] = ρ-x[3]
        jac[3,1] = x[2]
        jac[1,2] = σ+zero(x[1])
        jac[2,2] = -1.0+zero(x[1])
        jac[3,2] = x[1]
        jac[1,3] = zero(x[1])
        jac[2,3] = -x[1]
        jac[3,3] = -β+zero(x[1])
        nothing
    end

    q0 = [19.0, 20.0, 50.0] #the initial condition
    xi = set_variables("δ", order=1, numvars=length(q0))

    tv1, lv1, xv1 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);

    tv2, lv2, xv2 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);

    @test tv1 == tv2
    @test lv1 == lv2
    @test xv1 == xv2

    tv1, lv1, xv1 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000, parse_eqs=false);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000, parse_eqs=false);

    tv2, lv2, xv2 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000,  parse_eqs=true);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000,  parse_eqs=true);

    @test tv1 == tv2
    @test lv1 == lv2
    @test xv1 == xv2
end
@testset "Lyapunov spectrum and `@taylorize` with `local` constants" begin

    #Lorenz system ODE:
    @taylorize function lorenz!(t, x, dx)
        #Lorenz system parameters
        local σ = 16.0
        local β = 4.0
        local ρ = 45.92

        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end
    # @show(methods(TaylorIntegration.jetcoeffs!))

    #Lorenz system Jacobian (in-place):
    function lorenz_jac!(jac, t, x)
        #Lorenz system parameters
        local σ = 16.0
        local β = 4.0
        local ρ = 45.92

        jac[1,1] = -σ+zero(x[1])
        jac[2,1] = ρ-x[3]
        jac[3,1] = x[2]
        jac[1,2] = σ+zero(x[1])
        jac[2,2] = -1.0+zero(x[1])
        jac[3,2] = x[1]
        jac[1,3] = zero(x[1])
        jac[2,3] = -x[1]
        jac[3,3] = -β+zero(x[1])
        nothing
    end

    q0 = [19.0, 20.0, 50.0] #the initial condition
    xi = set_variables("δ", order=1, numvars=length(q0))

    tv1, lv1, xv1 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);

    tv2, lv2, xv2 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);

    @test tv1 == tv2
    @test lv1 == lv2
    @test xv1 == xv2

    tv1, lv1, xv1 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000, parse_eqs=false);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000, parse_eqs=false);

    tv2, lv2, xv2 = lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000,  parse_eqs=true);
    @time lyap_taylorinteg(lorenz!, q0, t0, tf, _order, _abstol,
        lorenz_jac!, maxsteps=2000,  parse_eqs=true);

    @test tv1 == tv2
    @test lv1 == lv2
    @test xv1 == xv2
end


@testset "Tests for throwing erros" begin
    ex = :(function f_p!(t, x, dx, y)
        dx[1] = x[2]
        dx[2] = -sin( x[1] )
        nothing  # `return` is needed at the end, for the vectorial case
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    ex = :(function f_p!(t, x)
        true && x
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    ex = :(function f_p!(t, x)
        "a"
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    ex = :(begin
        x=1
        x+x
    end)
    @test_throws KeyError TaylorIntegration._make_parsed_jetcoeffs(ex)
end
