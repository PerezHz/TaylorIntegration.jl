# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries: constant_term, _dict_unary_calls, _dict_binary_calls

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

using Espresso: subs, ExGraph, to_expr, sanitize, genname, rewrite




const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Taylor1{T}, __vT::Vector{T})
    order = __x.order
    __vT[1] = __t0
end)
);

const _LOOP_PARSEDFN_SCALAR = sanitize(:(
    for ord in 1:order
        ordnext = ord+1
    end
    # Symbol("_ret_var")
    )
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function parsed_jetcoeffs!{T<:Number}(__t0::T, __x::Vector{Taylor1{T}}, __dx::Vector{Taylor1{T}},
            __xaux::Vector{Taylor1{T}}, __vT::Vector{T})

    order = __x[1].order
    __vT[1] = __t0
end)
);

const _LOOP_PARSEDFN_VECTOR = sanitize(:(
    for ord in 1:order
        ordnext = ord+1
    end
    # nothing
    )
);




function _head_newfunct(fn, fnargs)

    # Construct common elements of the new expression
    if length(fnargs) == 2
        newfunction = copy(_HEAD_PARSEDFN_SCALAR)
    elseif length(fnargs) == 3
        newfunction = copy(_HEAD_PARSEDFN_VECTOR)
    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    # Rename the new function
    # newfunction.args[1].args[1].args[1] = Symbol(fn, "_parsed_jetcoeffs!")
    newfunction = subs( newfunction,
        Dict(:(parsed_jetcoeffs!) => Symbol(fn, "_parsed_jetcoeffs!")) )


    # Adapt body to insert new lines
    # newfunction.args[2].args[3].args[2] = Expr(:block, ordnext)
    # newfunction = subs(newfunction,
    #     Dict(:(ordnext = ord+1) => Expr(:block, :(ordnext = ord+1))))

    newfunction
end


function _replace_exprs!(aaa::Expr, aavar::Symbol, v_vars)

    if length(aaa.args) == 2

        # Unary call
        @assert(in(aaa.args[1], keys(_dict_unary_calls)))
        fnexpr, aux_fnexpr = _dict_unary_calls[aaa.args[1]]
        fnexpr = subs(fnexpr, Dict(:_res => aavar,
            :_arg1 => aaa.args[2], :_k => :ord))
        # Auxiliary expression
        if aux_fnexpr.head != :nothing
            newvar = genname()
            aux_fnexpr = subs(aux_fnexpr,
                Dict(:_arg1 => aaa.args[2], :_aux => newvar))
            push!(v_vars, newvar)
            fnexpr = subs(fnexpr, Dict(:_aux => newvar))
        end
    elseif length(aaa.args) == 3

        # Binary call; no auxiliary expressions needed
        @assert(in(aaa.args[1], keys(_dict_binary_calls)))
        fnexpr, aux_fnexpr = _dict_binary_calls[aaa.args[1]]
        fnexpr = subs(fnexpr, Dict(:_res => aavar,
            :_arg1 => aaa.args[2], :_arg2 => aaa.args[3],
            :_k => :ord))
    else
        throw(ArgumentError(":call is not unary or binary"))
    end

    push!(v_vars, aavar)

    return fnexpr, aux_fnexpr
end


function _preamble_body(fnbody, fnargs)

    # Unfolds the body to the expresion graph (AST) of the function,
    # a priori as unary and binary calls
    fnbody = sanitize(to_expr( ExGraph(fnbody) ))

    v_vars = Symbol[]            # List of symbols with created variables
    v_aux = Dict{Symbol,Expr}()  # Auxiliary functions
    v_assign = Tuple{Int,Expr}[] # Numeric assignements (to be deleated)

    # Populate v_vars, v_assign, v_aux
    for (i,aa) in enumerate(fnbody.args)
        aavar = aa.args[1]
        if isa(aa.args[2], Expr)
            aaa = aa.args[2]
            fnexpr, auxfnexpr = _replace_exprs!(aaa, aavar, v_vars)
            fnbody.args[i] = fnexpr
            if auxfnexpr.head != :nothing
                newvar = v_vars[end]
                push!(v_aux, auxfnexpr.args[1] => auxfnexpr.args[2])
            end
        elseif isa(aa.args[2], Number)
            push!(v_assign, (i,aa))
        else
        end
    end

    # Clean-up numeric assignements
    for kk in reverse(eachindex(v_assign))
        deleteat!(fnbody.args, v_assign[kk][1])
        fnbody = subs(fnbody,
            Dict(v_assign[kk][2].args[1] => v_assign[kk][2].args[2]))
    end

    # Define temporary variables
    preamble = Expr(:block,)
    for i in eachindex(v_vars)
        nv = v_vars[i]
        if in(nv, keys(v_aux))
            push!(preamble.args, parse("$nv = Taylor1( $(v_aux[nv]) , order)") )
            continue
        end
        push!(preamble.args, parse("$nv = zero( $(fnargs[2]) )") )
    end

    return preamble, fnbody
end



function to_parsed_jetcoeffs( ex )
    # Capture the body of the passed function
    @capture( shortdef(ex), fn_(fnargs__) = fnbody_ ) ||
        throw(ArgumentError("Must be a function call\n", ex))

    # Set up new function
    newfunction = _head_newfunct(fn, fnargs)

    # Block for th for-loop; it will include parsed body
    forloopblock = copy(_LOOP_PARSEDFN_SCALAR)
    forloopblock = subs(forloopblock,
        Dict(:(ordnext = ord + 1) => Expr(:block, :(ordnext = ord + 1))) )

    # Transform graph representation of the body of the function
    preamble, fnbody = _preamble_body(fnbody, fnargs);

    #### FALTA INCLUIR LA RECURRENCIA EN LAS ECS!!

    # Guessed return variable
    retvar = preamble.args[end].args[1]

    # Add preamble to newfunction
    push!(newfunction.args[2].args, preamble.args...)

    # Add parsed fnbody to forloopblock
    push!(forloopblock.args[2].args, fnbody.args...)

    # Push preamble and forloopblock to newfunction
    push!(newfunction.args[2].args, forloopblock);

    if length(fnargs) == 2
        push!(newfunction.args[2].args, parse("return $retvar"))
    else
        push!(newfunction.args[2].args, parse("return nothing"))
    end

    # Rename variables of the body of the new function
    newfunction = subs(newfunction,
        Dict(fnargs[1] => :(__vT), fnargs[2] => :(__x)))
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(fnargs[3] => :(__dx)))
    end

    newfunction
end
