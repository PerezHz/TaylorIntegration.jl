# ==========
# Native interface (`taylorinteg`) vs. common interface (`solve(prob, TaylorMethod(order))`)
# ==========
# Kepler problem, parsed, integrated for 0.5, 1, 2 and 5 periods with each interface. From the times
# and the numbers of steps, `compare.jl` fits for each interface the fixed cost per call `a` and
# the cost per step `b` (time = a + b * steps, least squares), and the ratios common/native.
# The group is defined only if the packages needed by the extension are available.

const HAS_COMMON_INTERFACE = try
    @eval using OrdinaryDiffEqCore
    @eval using SciMLBase
    true
catch
    @warn "OrdinaryDiffEqCore / SciMLBase not available; skipping the Interface benchmarks"
    false
end

if HAS_COMMON_INTERFACE
    let
        q0, μ = KEPLER_Q0, KEPLER_MU
        order, abstol, maxsteps = KEPLER_ORDER, KEPLER_ABSTOL, KEPLER_MAXSTEPS
        for periods in (0.5, 1.0, 2.0, 5.0)
            tf, label = 2π * periods, string(periods)
            prob = SciMLBase.ODEProblem(kepler1!, q0, (0.0, tf), μ)
            nnative = nsteps(taylorinteg(kepler1!, q0, 0.0, tf, order, abstol, μ; maxsteps, dense = false))
            ncommon = nsteps(SciMLBase.solve(prob, TaylorMethod(order); abstol, maxiters = maxsteps))
            add!("Interface", "native/$label", @benchmarkable(
                taylorinteg(kepler1!, $q0, 0.0, $tf, $order, $abstol, $μ; maxsteps = $maxsteps, dense = false)),
                steps = nnative)
            add!("Interface", "common/$label", @benchmarkable(
                SciMLBase.solve($prob, TaylorMethod($order); abstol = $abstol, maxiters = $maxsteps)),
                steps = ncommon)
        end
    end
end