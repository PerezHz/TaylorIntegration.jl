#Lyapunov Benchmarks

@taylorize function lorenz!(dx,x,param,t)
    σ, ρ, β = param
    dx[1] = σ*(x[2]-x[1])
    dx[2] = x[1]*(ρ-x[3])-x[2]
    dx[3] = x[1]*x[2]-β*x[3]
    nothing
end

SUITE = BenchmarkGroup()

# const TI = TaylorIntegration
const _abstol = 1.0e-20
const _order = 28
const pars = [16.0, 45.92, 4.0]
const t0 = 0.0
const tf1 = 1000.0
const maxsteps1 = 500_000
const tf2 = 10_000.0
const maxsteps2 = 50_000_000
const tf3 = 100_000.0
const maxsteps3 = 500_000_000
q0 = [19.0, 20.0, 50.0]
set_variables("δ", order=1, numvars=3)


SUITE["Lorenzlyap"] = BenchmarkGroup()
    
SUITE["Lorenzlyap"]["lorenzlyap1-1"] = @benchmarkable lyap_taylorinteg(
        lorenz!, $q0, $t0, $tf1, $_order, $_abstol, $pars, maxsteps=$maxsteps1)
SUITE["Lorenzlyap"]["lorenzlyap1-2"] = @benchmarkable lyap_taylorinteg(
        lorenz!, $q0, $t0, $tf2, $_order, $_abstol, $pars, maxsteps=$maxsteps1)
SUITE["Lorenzlyap"]["Lorenzlyap1-3"] = @benchmarkable lyap_taylorinteg(
        lorenz!, $q0, $t0, $tf3, $_order, $_abstol, $pars, maxsteps=$maxsteps1)



