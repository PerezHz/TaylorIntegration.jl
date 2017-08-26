# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries, TaylorIntegration
using Base.Test

const _order = 28
const _abstol = 1.0E-20

eqs1(t,z) = -z
eqs2(t,z) = im*z
function eqs3!(t, z, Dz)
    Dz[1] = eqs1(t,z[1])
    Dz[2] = eqs2(t,z[2])
    nothing
end

@testset "Test integration of ODE with complex dependent variables (1)" begin
    z0 = complex(0.0, 1.0)
    tr = 0.0:pi/8:2pi

    zsol1 = taylorinteg(eqs1, z0, tr, 28, 1.0e-20, maxsteps=1)
    @test length(zsol1) == length(tr)
    zsol1 = taylorinteg(eqs1, z0, tr, 28, 1.0e-20)
    @test zsol1[1] == z0
    @test isapprox( zsol1[2]  , z0*exp(-tr[2]) )
    @test isapprox( zsol1[6], z0*exp(-tr[6]) )
    @test isapprox( zsol1[end], z0*exp(-tr[end]) )
    tt, zsol1 = taylorinteg(eqs1, z0, 0.0, 2pi, 28, 1.0e-20)
    @test zsol1[1] == z0
    @test isapprox( zsol1[2]  , z0*exp(-tt[2]) )
    @test isapprox( zsol1[end], z0*exp(-tt[end]) )
end

@testset "Test integration of ODE with complex dependent variables (2)" begin
    z0 = complex(0.0, 1.0)
    tr = 0.0:pi/8:2pi

    zsol2 = taylorinteg(eqs2, z0, tr, 28, 1.0e-20, maxsteps=1)
    @test length(zsol2) == length(tr)
    zsol2 = taylorinteg(eqs2, z0, tr, 28, 1.0e-20)
    @test zsol2[1] == z0
    @test isapprox( zsol2[3], z0*exp( complex(0.0, tr[3])) )
    @test isapprox( zsol2[5], z0*exp( complex(0.0, tr[5])) )
    tt, zsol2 = taylorinteg(eqs2, z0, 0.0, 2pi, 28, 1.0e-20)
    @test zsol2[1] == z0
    @test isapprox( zsol2[2], z0*exp( complex(0.0, tt[2])) )
    @test isapprox( zsol2[end], z0*exp( complex(0.0, tt[end])) )
end

@testset "Test integration of ODE with complex dependent variables (3)" begin
    z0 = complex(0.0, 1.0)
    zz0 = [z0, z0]
    tr = 0.0:pi/8:2pi

    zsol3 = taylorinteg(eqs3!, zz0, tr, 28, 1.0e-20, maxsteps=1)
    @test size(zsol3) == (length(tr), length(zz0))
    zsol3 = taylorinteg(eqs3!, [z0, z0], tr, 28, 1.0e-20)
    @test isapprox( zsol3[4,1], z0*exp(-tr[4]) )
    @test isapprox( zsol3[7,1], z0*exp(-tr[7]) )
    @test isapprox( zsol3[4,2], z0*exp( complex(0.0, tr[4])) )
    @test isapprox( zsol3[7,2], z0*exp( complex(0.0, tr[7])) )
    tt, zsol3 = taylorinteg(eqs3!, zz0, 0.0, 2pi, 28, 1.0e-20)
    @test zsol3[1,1] == z0
    @test zsol3[1,2] == z0
    @test isapprox( zsol3[2,1], z0*exp( -tt[2]) )
    @test isapprox( zsol3[end,1], z0*exp( -tt[end]) )
    @test isapprox( zsol3[2,2], z0*exp( complex(0.0, tt[2])) )
    @test isapprox( zsol3[end,2], z0*exp( complex(0.0, tt[end])) )
end
