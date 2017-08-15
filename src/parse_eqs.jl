# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of MacroTools and Espresso
using MacroTools: @capture, shortdef

using Espresso: subs, ExGraph, to_expr, sanitize, genname, isindexed, get_vars



# Define some constants to create the newly (parsed) functions
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Taylor1{T})

    order = __x.order
    __tT = Taylor1([__t0, one(T)], order)
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Vector{Taylor1{T}},
        __dx::Vector{Taylor1{T}})

    order = __x[1].order
    __tT = Taylor1([__t0, one(T)], order)
end)
);

const _LOOP_PARSEDFN = sanitize(:(
    for ord in 1:order-1
        ordnext = ord+1
    end
    )
);


"""
`_extract_parts(ex::Expr)`

Returns the function name, the function arguments, and the body of
a function passed as an `Expr`. The function may be provided as
a one-line function, or in the long form (anonymous functions
do not work).

"""
function _extract_parts(ex, debug=false)

    # Capture name, args and body
    @capture( shortdef(ex), fn_(fnargs__) = fnbody_ ) ||
        throw(ArgumentError("Must be a function call\n", ex))

    # Standarize fnbody, same structure for one-line of long form functions
    if length(fnbody.args) > 1
        fnbody.args[1] = Expr(:block, copy(fnbody.args)...)
        deleteat!(fnbody.args, 2:length(fnbody.args))
    end

    # Special case: the last arg of the function is a simple
    # assignement (symbol) or a numerical value
    if isa(fnbody.args[1].args[end], Symbol)
        fnbody.args[1].args[end] = :( identity($(fnbody.args[1].args[end])) )
    elseif isa(fnbody.args[1].args[end], Number)
        fnbody.args[1].args[end] =
            :( $(fnbody.args[1].args[end])+zero($(fnargs[1])) )
    end

    if debug
        @show(fn, fnargs, fnbody)
        println()
    end

    return fn, fnargs, fnbody
end


"""
`_newhead(fn, fnargs)`

Creates the head of the new function, whose name is
`fn` appended by `_parsed_jetcoeffs!`. Here, `fn`
is the Symbol that represents the original name of the
function, and `fnargs` is a vector with the
arguments of the function (two or three).

"""
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
        Dict(:(parsed_jetcoeffs!) => Symbol(fn, "_parsed_jetcoeffs!")) )

    return newfunction
end


"""
`_replacecalls!(fnold, newvar, v_vars)`

Replaces the symbols of unary and binary calls
of the expression `fnold` which defines `newvar`
by their mutating functions in TaylorSeries.jl.
The vector `v_vars` is updated with the new auxiliary
variables introduced.

"""
function _replacecalls!(fnold::Expr, newvar::Symbol, v_vars::Vector{Symbol})

    ll = length(fnold.args)

    if ll == 2
        # Unary call
        @assert(in(fnold.args[1], keys(TaylorSeries._dict_unary_calls)))

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_unary_calls[fnold.args[1]]
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
        @assert(in(fnold.args[1], keys(TaylorSeries._dict_binary_calls)))

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_binary_calls[fnold.args[1]]
        fnexpr = subs(fnexpr, Dict(:_res => newvar,
            :_arg1 => fnold.args[2], :_arg2 => fnold.args[3],
            :_k => :ord))
        def_fnexpr = subs(def_fnexpr, Dict(:_res => newvar,
            :_arg1 => :(constant_term($(fnold.args[2]))),
            :_arg2 => :(constant_term($(fnold.args[3]))),
            :_k => :ord))
    else
        throw(ArgumentError("""
            A call in the function definition is not unary or binary;
            use parenthesis to have only binary and unary operations!"""
            ))
    end

    return fnexpr, def_fnexpr, aux_fnexpr
end


"""
`_newpreamble(v_vars, v_preamb, fnargs)`

Returns the preamble expression, where the auxiliary `Taylor1` objects
are defined.

"""
function _newpreamble(v_vars, v_preamb, fnargs)
    preamble = Expr(:block,)
    for i in eachindex(v_vars)
        nv = v_vars[i]
        if in(nv, keys(v_preamb))
            push!(preamble.args, parse("$nv = Taylor1( $(v_preamb[nv]) , order)") )
        else
            # Is this unreachable ? Do we really need fnargs ?
            throw(ArgumentError("_newpreamble: $nv"))
            push!(preamble.args, parse("$nv = zero( $(fnargs[2]) )") )
        end
    end
    return preamble
end


"""
`_preamble_body(fnbody, fnargs, debug=false)`

Returns the preamble, the body and the guessed returned
variable, which will be used to build the parsed
function. `fnbody` is the expression with the body of
the original function, `fnargs` is a vector of symbols
of the original diferential equations function.

"""
function _preamble_body(fnbody, fnargs, debug=false)

    # Bookkeeping
    v_vars = Symbol[]               # List of symbols with created variables
    v_preamb = Dict{Symbol,Expr}()  # Auxiliary definitions
    v_assign = Tuple{Int,Expr}[]    # Numeric assignments (to be deleted)

    # Rename vars to have the body in non-indexed form
    fnbody, v_indexed, d_indx = _rename_indexedvars(fnbody)
    debug && (@show(fnbody, v_indexed, d_indx); println())

    # Unfolds the body to the expresion graph (AST) of the function,
    # a priori as unary and binary calls
    # newfnbody = sanitize(to_expr( ExGraph(fnbody) ))
    newfnbody = Expr(:block,)
    for (i,ex) in enumerate(fnbody.args[1].args)
        if isa(ex, Expr)
            ex.head == :line && continue
            nex =  to_expr(ExGraph(ex))
            push!(newfnbody.args, nex.args[2:end]...)
        else
            @show(typeof(ex))
            throw(ArgumentError(ex, "is not an `Expr`"))
            #
        end
    end

    debug && @show(newfnbody)

    # Needed, if `newfnbody` consists of a single assignment (unary call)
    if newfnbody.head == :(=)
        newfnbody = Expr(:block, newfnbody)
        debug && @show(newfnbody)
    end

    debug && (@show(newfnbody); println())

    # Populate v_vars, v_assign, v_preamb
    for (i, aa) in enumerate(newfnbody.args)
        aavar = aa.args[1]
        aaa = aa.args[2]
        if isa(aaa, Expr)
            fnexpr, def_fnexpr, auxfnexpr = _replacecalls!(aaa, aavar, v_vars)
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
            fnexpr, def_fnexpr, auxfnexpr = _replacecalls!(bbb, bbvar, v_vars)
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

    debug && @show(v_vars, v_assign, v_preamb)

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

    # Bring back indexed variables into place
    preamble = subs( preamble, d_indx)
    newfnbody = subs( newfnbody, d_indx)
    retvar = subs( retvar, d_indx)

    return preamble, newfnbody, retvar
end


"""
`_rename_indexedvars(fnbody)`

Renames the indexed variables (using `Espresso.genname()`) that
exists in `fnbody`. Returns `fnbody` with the renamed variables,
a vector of symbols with the new variables, and a dictionary
that links the new variables to the old indexed variables.

"""
function _rename_indexedvars(fnbody)

    v_indexed = Symbol[]
    d_indx = Dict{Symbol, Expr}()

    !(isindexed(fnbody)) && return fnbody, v_indexed, d_indx

    # Obtain variables
    vvars = get_vars(fnbody, rec=true)

    # Rename indexed variables
    for v in vvars
        if isindexed(v)
            newvar = genname()
            push!(v_indexed, newvar)
            push!(d_indx, newvar => v)
            fnbody = subs(fnbody, Dict(v => newvar))
            vvars .= subs.(vvars, Dict(v => newvar))
        end
    end

    return fnbody, v_indexed, d_indx
end


"""
`_make_parsed_jetcoeffs( ex, debug=false )`

This function constructs a new function, equivalent to the
differential equations, which exploits the mutating functions
of TaylorSeries.jl.

"""
function _make_parsed_jetcoeffs( ex, debug=false )

    # Extract the name, args and body of the function
    fn, fnargs, fnbody = _extract_parts(ex, debug)

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
