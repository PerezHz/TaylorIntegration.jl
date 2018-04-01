using TaylorIntegration, Base.Test

@testset "Test integration of ODE with numbers in common interface" begin
    f(u,p,t) = u
    u0 = 0.5
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob,TaylorMethod(50),abstol=1e-20)

    @test sol[end] - 0.5exp(1) < 1e-12
end

@testset "Test integration of ODE with abstract arrays in common interface" begin
    f(du,u,p,t) = (du .= u)
    u0 = rand(4,2)
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob,TaylorMethod(50),abstol=1e-20)

    @test norm(sol[end] - u0.*exp(1)) < 1e-12
end
