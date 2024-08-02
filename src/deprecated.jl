function taylorinteg_old(f, x0, t0, tmax, order, abstol, params; kwargs...)
    sol = taylorinteg(f, x0, t0, tmax, order, abstol, params; kwargs...)
    if kwargs[:dense]
        return sol.t, sol.x, sol.p
    else
        return sol.t, sol.x
    end
end

function taylorinteg_old(f, g, x0, t0, tmax, order, abstol, params; kwargs...)
    sol = taylorinteg(f, g, x0, t0, tmax, order, abstol, params; kwargs...)
    if kwargs[:dense]
        return sol.t, sol.x, sol.p, sol.tevents, sol.xevents, sol.gresids
    else
        return sol.t, sol.x, sol.tevents, sol.xevents, sol.gresids
    end
end

Base.@deprecate taylorinteg(f, x0, t0, tmax, order, abstol, ::Val{false}, params; maxsteps=500, parse_eqs=true) taylorinteg_old(f, x0, t0, tmax, order, abstol, params; maxsteps=500, parse_eqs=true, dense=false)
Base.@deprecate taylorinteg(f, x0, t0, tmax, order, abstol, ::Val{true}, params; maxsteps=500, parse_eqs=true) taylorinteg_old(f, x0, t0, tmax, order, abstol, params; maxsteps=500, parse_eqs=true, dense=true)
Base.@deprecate taylorinteg(f, g, x0, t0, tmax, order, abstol, ::Val{false}, params; maxsteps=500, parse_eqs=true, eventorder=0, newtoniter=10, nrabstol=eps(typeof(t0))) taylorinteg_old(f, g, x0, t0, tmax, order, abstol, params; maxsteps=500, parse_eqs=true, dense=false, eventorder=0, newtoniter=10, nrabstol=eps(typeof(t0)))
Base.@deprecate taylorinteg(f, g, x0, t0, tmax, order, abstol, ::Val{true}, params; maxsteps=500, parse_eqs=true, eventorder=0, newtoniter=10, nrabstol=eps(typeof(t0))) taylorinteg_old(f, g, x0, t0, tmax, order, abstol, params; maxsteps=500, parse_eqs=true, dense=true, eventorder=0, newtoniter=10, nrabstol=eps(typeof(t0)))
