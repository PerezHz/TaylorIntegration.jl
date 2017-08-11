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
    __tT = Taylor1(__vT, order)
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Vector{Taylor1{T}},
        __dx::Vector{Taylor1{T}},
        __xaux::Vector{Taylor1{T}}, __vT::Vector{T})

    order = __x[1].order
    __vT[1] = __t0
    __tT = Taylor1(__vT, order)
end)
);

const _LOOP_PARSEDFN = sanitize(:(
    for ord in 1:order-1
        ordnext = ord+1
    end
    )
);


function _extract_parts(ex, debug=false)

    # Capture name, args and body
    @capture( shortdef(ex), fn_(fnargs__) = fnbody_ ) ||
        throw(ArgumentError("Must be a function call\n", ex))

    # Standarize fnbody
    if length(fnbody.args) > 1
        fnbody.args[1] = Expr(:block, copy(fnbody.args)...)
        deleteat!(fnbody.args, 2:length(fnbody.args))
    end

    # Sanity check: the function is a simple assignement or a numerical value
    if isa(fnbody.args[1].args[end], Symbol)
        fnbody.args[1].args[end] = :( identity($(fnbody.args[1].args[end])) )
        # debug && @show(fnbody)
    elseif isa(fnbody.args[1].args[end], Number)
        fnbody.args[1].args[end] =
            :( $(fnbody.args[1].args[end])+zero($(fnargs[1])) )
        # debug && @show(fnbody)
    end

    if debug
        @show(fn, fnargs, fnbody)
        println()
    end

    return fn, fnargs, fnbody
end


function _newhead(fn, fnargs)

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


function _newpreamble(v_vars, v_preamb, fnargs)
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


function _preamble_body(fnbody, fnargs, debug=false)

    # Bookkeeping
    v_vars = Symbol[]               # List of symbols with created variables
    v_preamb = Dict{Symbol,Expr}()  # Auxiliary definitions
    v_assign = Tuple{Int,Expr}[]    # Numeric assignments (to be deleted)

    # Unfolds the body to the expresion graph (AST) of the function,
    # a priori as unary and binary calls
    newfnbody = sanitize(to_expr( ExGraph(fnbody) ))

    # Needed, if `newfnbody` consists of a single assignment (unary call)
    if newfnbody.head == :(=)
        newfnbody = Expr(:block, newfnbody)
    end

    if debug
        @show(newfnbody)
        println()
    end

    # Populate v_vars, v_assign, v_preamb
    for (i, aa) in enumerate(newfnbody.args)
        aavar = aa.args[1]
        aaa = aa.args[2]
        if isa(aaa, Expr)
            fnexpr, def_fnexpr, auxfnexpr = _replacevars!(aaa, aavar, v_vars)
            push!(v_preamb, def_fnexpr.args[1] => def_fnexpr.args[2])
            newfnbody.args[i] = fnexpr
            if auxfnexpr.head != :nothing
                push!(v_preamb, auxfnexpr.args[1] => auxfnexpr.args[2])
            end
            push!(v_vars, aavar)
            #
        elseif isa(aaa, Symbol) # occurs when there is a simple assignment
            bb = subs(aa, Dict(aaa => :(identity($aaa))))
            bbvar = bb.args[1]
            bbb = bb.args[2]
            fnexpr, def_fnexpr, auxfnexpr = _replacevars!(bbb, bbvar, v_vars)
            push!(v_preamb, def_fnexpr.args[1] => def_fnexpr.args[2])
            newfnbody.args[i] = fnexpr
            if auxfnexpr.head != :nothing
                push!(v_preamb, auxfnexpr.args[1] => auxfnexpr.args[2])
            end
            push!(v_vars, bbvar)
            #
        elseif isa(aaa, Number)
            push!(v_assign, (i, aa))
            #
        else #needed?
            @show(aa.args[2], typeof(aa.args[2]))
            error("Different from `Expr`, `Symbol` or `Number`")
            #
        end
    end

    # Define premable (temporary allocations)
    preamble = _newpreamble(v_vars, v_preamb, fnargs)

    # Clean-up numeric assignments in preamble and body
    for kk in reverse(v_assign)
        newfnbody = subs(newfnbody, Dict(kk[2].args[1] => kk[2].args[2]))
        preamble = subs(preamble, Dict(kk[2].args[1] => kk[2].args[2]))
        deleteat!(newfnbody.args, kk[1])
    end

    # Check consistency of retvar
    @assert(v_vars[end] == preamble.args[end].args[1])

    # Guessed return variable; last included in v_vars/preamble
    # retvar = preamble.args[end].args[1]
    retvar = v_vars[end]

    return preamble, newfnbody, retvar
end


function _rename_indexedvars(fnbody)

    v_indexed = Symbol[]
    d_indx = Dict{Symbol, Expr}()

    !Espresso.isindexed(fnbody) && return fnbody, v_indexed, d_indx

    # Obtain variables
    vvars = Espresso.get_vars(fnbody, rec=true)

    # Rename indexed variables
    for v in vvars
        if Espresso.isindexed(v)
            newvar = Espresso.genname()
            push!(v_indexed, newvar)
            push!(d_indx, newvar => v)
            fnbody = subs(fnbody, Dict(v => newvar))
            vvars .= subs.(vvars, Dict(v => newvar))
        end
    end

    return fnbody, v_indexed, d_indx
end



function _to_parsed_jetcoeffs( ex, debug=false )

    # Extract the name, args and body of the function
    fn, fnargs, fnbody = _extract_parts(ex, debug)

    # Rename vars to have the body in non-indexed form
    fnbody, v_indexed, d_indx = _rename_indexedvars(fnbody)

    # Set up new function
    newfunction = _newhead(fn, fnargs)

    # for-loop block; it will include parsed body
    forloopblock = copy(_LOOP_PARSEDFN)
    forloopblock = subs(forloopblock,
        Dict(:(ordnext = ord + 1) => Expr(:block, :(ordnext = ord + 1))) )

    # Transform graph representation of the body of the function
    preamble, fnbody, retvar = _preamble_body(fnbody, fnargs, debug)

    if debug
        @show(preamble, fnbody, retvar)
        println()
    end

    # Recursion relation
    # @inbounds x[ordnext] = dx[ord]/ord
    if length(fnargs) == 2
        rec_preamb = :( $(fnargs[2])[2] = $(retvar)[1] )
        rec_fnbody = :( $(fnargs[2])[ordnext+1] =
            $(retvar)[ordnext]/ordnext )
    elseif length(fnargs) == 3
        rec_preamb = :( $(fnargs[2])[:][2] .= $(retvar)[:][1] )
        rec_fnbody = :( $(fnargs[2])[:][ordnext+1] .=
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
        Dict(fnargs[1] => :(__tT), fnargs[2] => :(__x)))
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(fnargs[3] => :(__dx)))
    end

    newfunction
end
