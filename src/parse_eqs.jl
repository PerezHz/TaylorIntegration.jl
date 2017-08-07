# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of TaylorSeries, MacroTools and Espresso
using TaylorSeries: _dict_unary_calls, _dict_binary_calls

# Simple access to all internal mutating functions from TaylorSeries
for kk in keys(_dict_unary_calls)
    args = _dict_unary_calls[kk][1].args[1]
    ex = :(using TaylorSeries: $(_dict_unary_calls[kk][1].args[1]) )
    eval(ex)
end
for kk in keys(_dict_binary_calls)
    args = _dict_binary_calls[kk][1].args[1]
    ex = :(using TaylorSeries: $(_dict_binary_calls[kk][1].args[1]) )
    eval(ex)
end

using MacroTools: @capture, shortdef

using Espresso: subs, ExGraph, to_expr, sanitize, genname



# Define some constants for the newly (parsed) functions
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Taylor1{T}, __vT::Vector{T})
    order = __x.order
    __vT[1] = __t0
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Vector{Taylor1{T}},
        __dx::Vector{Taylor1{T}},
        __xaux::Vector{Taylor1{T}}, __vT::Vector{T})

    order = __x[1].order
    __vT[1] = __t0
end)
);

const _LOOP_PARSEDFN = sanitize(:(
    for ord in 1:order-1
        ordnext = ord+1
    end
    )
);


function _nfnhead(fn, fnargs)

    # Construct common elements of the new expression
    if length(fnargs) == 2
        newfunction = copy(_HEAD_PARSEDFN_SCALAR)
    elseif length(fnargs) == 3
        newfunction = copy(_HEAD_PARSEDFN_VECTOR)
    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    # Rename the new function; equivalent to:
    # newfunction.args[1].args[1].args[1] = Symbol(fn, "_parsed_jetcoeffs!")
    newfunction = subs( newfunction,
        Dict(:(parsed_jetcoeffs!) => Symbol(fn, "_parsed!")) )

    return newfunction
end


function _replacevars!(fnold::Expr, newvar::Symbol, v_vars)

    ll = length(fnold.args)

    if ll == 2
        # Unary call
        @assert(in(fnold.args[1], keys(_dict_unary_calls)))

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = _dict_unary_calls[fnold.args[1]]
        fnexpr = subs(fnexpr, Dict(:_res => newvar,
            :_arg1 => fnold.args[2], :_k => :ord))
        def_fnexpr = subs(def_fnexpr, Dict(:_res => newvar,
            :_arg1 => :(constant_term($(fnold.args[2]))), :_k => :ord))

        # Auxiliary expression
        if aux_fnexpr.head != :nothing
            newvar = genname()
            aux_fnexpr = subs(aux_fnexpr,
                Dict(:_arg1 => :(constant_term($(fnold.args[2]))), :_aux => newvar))
            fnexpr = subs(fnexpr, Dict(:_aux => newvar))
            push!(v_vars, newvar)
        end
    elseif ll == 3
        # Binary call; no auxiliary expressions needed
        @assert(in(fnold.args[1], keys(_dict_binary_calls)))

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = _dict_binary_calls[fnold.args[1]]
        fnexpr = subs(fnexpr, Dict(:_res => newvar,
            :_arg1 => fnold.args[2], :_arg2 => fnold.args[3],
            :_k => :ord))
        def_fnexpr = subs(def_fnexpr, Dict(:_res => newvar,
            :_arg1 => :(constant_term($(fnold.args[2]))),
            :_arg2 => :(constant_term($(fnold.args[3]))),
            :_k => :ord))
    else
        throw(ArgumentError("`:call` is not unary or binary; use parenthesis!"))
    end

    return fnexpr, def_fnexpr, aux_fnexpr
end


function _nfnpreamble(v_vars, v_preamb, fnargs)
    preamble = Expr(:block,)
    for i in eachindex(v_vars)
        nv = v_vars[i]
        if in(nv, keys(v_preamb))
            push!(preamble.args, parse("$nv = Taylor1( $(v_preamb[nv]) , order)") )
            continue
        else
            # is this unreachable ?
            push!(preamble.args, parse("$nv = zero( $(fnargs[2]) )") )
        end
    end
    return preamble
end


function _nfnpreamble_body(fnbody, fnargs)

    # Unfolds the body to the expresion graph (AST) of the function,
    # a priori as unary and binary calls
    newfnbody = sanitize(to_expr( ExGraph(fnbody) ))

    v_vars = Symbol[]            # List of symbols with created variables
    v_preamb = Dict{Symbol,Expr}()  # Auxiliary definitions
    v_assign = Tuple{Int,Expr}[] # Numeric assignements (to be deleated)

    # Populate v_vars, v_assign, v_preamb
    for (i,aa) in enumerate(newfnbody.args)
        aavar = aa.args[1]
        aaa = aa.args[2]
        if isa(aaa, Expr)
            push!(v_vars, aavar)
            fnexpr, def_fnexpr, auxfnexpr = _replacevars!(aaa, aavar, v_vars)
            push!(v_preamb, def_fnexpr.args[1] => def_fnexpr.args[2])
            newfnbody.args[i] = fnexpr
            if auxfnexpr.head != :nothing
                newvar = v_vars[end]
                push!(v_preamb, auxfnexpr.args[1] => auxfnexpr.args[2])
            end
        elseif isa(aaa, Number)
            push!(v_assign, (i, aa))
        else #needed?
            @show(aa.args[2], typeof(aa.args[2]))
            error("Different from `Number` or `Expr`")
        end
    end

    # Define premable (temporary allocations)
    preamble = _nfnpreamble(v_vars, v_preamb, fnargs)

    # Clean-up numeric assignements in preamble and body
    for kk in reverse(v_assign)
        newfnbody = subs(newfnbody, Dict(kk[2].args[1] => kk[2].args[2]))
        preamble = subs(preamble, Dict(kk[2].args[1] => kk[2].args[2]))
        deleteat!(newfnbody.args, kk[1])
    end

    return preamble, newfnbody
end


function _to_parsed_jetcoeffs( ex )
    # Capture the body of the passed function
    @capture( shortdef(ex), fn_(fnargs__) = fnbody_ ) ||
        throw(ArgumentError("Must be a function call\n", ex))

    # Set up new function
    newfunction = _nfnhead(fn, fnargs)

    # for-loop block; it will include parsed body
    forloopblock = copy(_LOOP_PARSEDFN)
    forloopblock = subs(forloopblock,
        Dict(:(ordnext = ord + 1) => Expr(:block, :(ordnext = ord + 1))) )

    # Transform graph representation of the body of the function
    preamble, fnbody = _nfnpreamble_body(fnbody, fnargs);

    # Guessed return variable
    retvar = preamble.args[end].args[1]

    # Recursion relation
    # @inbounds x[ordnext] = dx[ord]/ord
    if length(fnargs) == 2
        rec_preamb = :( $(fnargs[2])[2] = $(retvar)[1] )
        rec_fnbody = :( $(fnargs[2])[ordnext+1] =
            $(retvar)[ordnext]/ordnext )
    elseif length(fnargs) == 3
        rec_preamb = :( $(fnargs[2:end])[2] .= $(retvar)[:][1] )
        rec_fnbody = :( $(fnargs[2:end])[ordnext+1] .=
            $(retvar)[:][ordnext]/ordnext )
    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    # Add preamble to newfunction
    push!(newfunction.args[2].args, preamble.args..., rec_preamb)

    # Add parsed fnbody to forloopblock
    push!(forloopblock.args[2].args, fnbody.args..., rec_fnbody)

    # Push preamble and forloopblock to newfunction
    push!(newfunction.args[2].args, forloopblock);
    push!(newfunction.args[2].args, parse("return nothing"))

    # Rename variables of the body of the new function
    newfunction = subs(newfunction,
        Dict(fnargs[1] => :(__vT), fnargs[2] => :(__x)))
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(fnargs[3] => :(__dx)))
    end

    newfunction
end
