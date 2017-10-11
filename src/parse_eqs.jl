# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of MacroTools and Espresso
using MacroTools: @capture, shortdef

using Espresso: subs, ExGraph, to_expr, sanitize, genname,
    find_vars, findex, find_indices, isindexed


# Define some constants to create the newly (parsed) functions
# Teh (irrelevant) `nothing` is there to have a block (:block); deleted later
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function jetcoeffs!{T<:Number, S<:Number}(::Type{Val{__fn}}, __tT::Taylor1{T},
        __x::Taylor1{S})

    order = __x.order
    nothing
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function jetcoeffs!{T<:Number, S<:Number}(::Type{Val{__fn}}, __tT::Taylor1{T},
        __x::Vector{Taylor1{S}}, __dx::Vector{Taylor1{S}})

    order = __x[1].order
    nothing
end)
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
        throw(ArgumentError("It must be a function call:\n $ex"))

    # Standarize fnbody, same structure for one-line of long form functions
    if length(fnbody.args) > 1
        fnbody.args[1] = Expr(:block, copy(fnbody.args)...)
        deleteat!(fnbody.args, 2:length(fnbody.args))
    end

    # Special case: the last arg of the function is a simple
    # assignement (symbol) or a numerical value
    if isa(fnbody.args[1].args[end], Symbol)
        # This accouns the case of having `nothing` at the end of the function
        if fnbody.args[1].args[end] == :nothing
            pop!(fnbody.args[1].args)
        else
            fnbody.args[1].args[end] = :(identity($(fnbody.args[1].args[end])))
        end
    elseif isa(fnbody.args[1].args[end], Number)
        fnbody.args[1].args[end] =
            :( $(fnbody.args[1].args[end])+zero($(fnargs[1])) )
    end

    debug && (@show(fn, fnargs, fnbody); println())

    return fn, fnargs, fnbody
end


"""
`_newhead(fn, fnargs)`

Creates the head of the new method of `jetcoeffs!`. Here,
`fn` is the name of the passed function and `fnargs` is a
vector with its arguments (which are two or three).

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

    # Add `TaylorIntegration` to create a new method of `jetcoeffs!`
    newfunction.args[1].args[1].args[1] =
        Expr(:., :TaylorIntegration, :(:jetcoeffs!))

    # Delete irrelevant `nothing`
    pop!(newfunction.args[2].args)

    return newfunction
end


"""
`_replacecalls!(fnold, newvar, d_indx, v_vars)`

Replaces the symbols of unary and binary calls
of the expression `fnold` which defines `newvar`
by their mutating functions in TaylorSeries.jl.
The vector `v_vars` is updated with the new auxiliary
variables introduced.

"""
function _replacecalls!(fnold::Expr, newvar::Symbol, d_indx, v_vars)

    ll = length(fnold.args)
    dcall = fnold.args[1]
    newarg1 = fnold.args[2]

    if ll == 2
        # Unary call
        @assert(in(dcall, keys(TaylorSeries._dict_unary_calls)))

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_unary_calls[dcall]
        fnexpr = subs(fnexpr, Dict(:_res => newvar,
            :_arg1 => newarg1, :_k => :ord))

        # if in(newvar, keys(d_indx)) || in(newarg1, keys(d_indx))
        def_fnexpr = :( _res = Taylor1($(def_fnexpr.args[2]), order) )
        # end
        def_fnexpr = subs(def_fnexpr, Dict(:_res => newvar,
            :_arg1 => :(constant_term($(newarg1))), :_k => :ord))

        # Auxiliary expression
        if aux_fnexpr.head != :nothing
            newaux = genname()
            # if in(newvar, keys(d_indx)) || in(newarg1, keys(d_indx))
            aux_fnexpr = :( _res = Taylor1($(aux_fnexpr.args[2]), order) )
            # end
            aux_fnexpr = subs(aux_fnexpr, Dict(:_res => newaux,
                :_arg1 => :(constant_term($(newarg1))), :_aux => newaux))
            fnexpr = subs(fnexpr, Dict(:_aux => newaux))
            !in(newaux, v_vars) && push!(v_vars, newaux)
        end
    elseif ll == 3
        # Binary call; no auxiliary expressions needed
        @assert(in(dcall, keys(TaylorSeries._dict_binary_calls)))
        newarg2 = fnold.args[3]

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_binary_calls[dcall]
        fnexpr = subs(fnexpr, Dict(:_res => newvar,
            :_arg1 => newarg1, :_arg2 => newarg2, :_k => :ord))

        def_fnexpr = :(_res = Taylor1($(def_fnexpr.args[2]), order) )
        def_fnexpr = subs(def_fnexpr, Dict(:_res => newvar,
            :_arg1 => :(constant_term($(newarg1))),
            :_arg2 => :(constant_term($(newarg2))),
            :_k => :ord))
    else
        throw(ArgumentError("""
            A call in the function definition is not unary or binary;
            use parenthesis to have only binary and unary operations!"""
            ))
    end

    # Bring back indexed variables
    fnexpr = subs(fnexpr, d_indx)
    def_fnexpr = subs(def_fnexpr, d_indx)
    aux_fnexpr = subs(aux_fnexpr, d_indx)

    return fnexpr, def_fnexpr, aux_fnexpr
end


"""
`_indexed_definitions!(preamble, d_indx, fnargs)`

Returns a vector with the definitions of some indexed auxiliary variables;
it may modify `d_indx` if new variables are introduced.

"""
function _indexed_definitions!(preamble::Expr, d_indx, fnargs, inloop=false)
    defspreamble = Expr[]

    for (i,ex) in enumerate(preamble.args)
        if isa(ex, Expr)
            # Ignore the following cases
            (ex.head == :line || ex.head == :return) && continue

            if (ex.head == :block)
                newblock = _indexed_definitions!(ex, d_indx, fnargs, inloop)
                append!(defspreamble, newblock)
            elseif (ex.head == :for)
                # indx = ex.args[1].args[1]
                # println("0 `for:`")
                loopbody = _indexed_definitions!(ex.args[2], d_indx, fnargs, true)
                # println("1 `for:`")
                append!(defspreamble, loopbody)
            else     # `ex.head` should be a :(:=) of some type
                # @show(ex)
                ex = subs(ex, d_indx)
                (inloop && isindexed(ex)) || continue
                # @show(ex)
                alhs = ex.args[1]
                arhs = ex.args[2]
                if isindexed(alhs)
                    (in(alhs.args[1], fnargs) || in(alhs, d_indx)) &&
                        continue
                    throw(ArgumentError(
                        "Indexed expression case not yet implemented $ex"))
                end
                newvar = genname()
                vars_indexed = findex(:(_X[_i...]), arhs)
                # Note: Is considering the first indexed var of arhs enough?
                var1 = vars_indexed[1].args[1]
                indx1 = vars_indexed[1].args[2]
                push!(d_indx, alhs => :($newvar[$indx1]) )
                exx = :($newvar = Array{Taylor1{S}}(length($var1)))
                push!(defspreamble, exx)
                ex.args[1] = :($newvar[$indx1])
                # @show(ex)
            end
        else
            @show(typeof(ex))
            throw(ArgumentError("$ex is not an `Expr`"))
            #
        end
    end
    return defspreamble
end


"""
`_newfnbody(fnbody)`

Returns a new (modified) body of the function, a priori unfolding
the expression graph (AST) as unary and binary calls.
"""
function _newfnbody(fnbody)
    # `fnbody` is assumed to be a `:block` `Expr`
    newfnbody = Expr(:block,)

    for (i,ex) in enumerate(fnbody.args)
        if isa(ex, Expr)
            # Ignore the following cases
            (ex.head == :line || ex.head == :return) && continue

            # Treat `for` loops separately
            if (ex.head == :block)
                newblock = _newfnbody(ex)
                push!(newfnbody.args, newblock )
            elseif (ex.head == :for)
                loopbody = _newfnbody( ex.args[2] )
                push!(newfnbody.args, Expr(:for, ex.args[1], loopbody))
            else
                nex = to_expr(ExGraph(ex))
                push!(newfnbody.args, nex.args[2:end]...)
            end
        else
            @show(typeof(ex))
            throw(ArgumentError("$ex is not an `Expr`"))
            #
        end
    end

    return newfnbody
end


"""
`_populate!(ex, preex, v_vars, v_assign, v_preamb, d_indx)`

Updates inplace the variables (`v_vars`), assignments (`v_assign`)
and the variables in the preamble (`v_preamb`) from the
expression `ex`, which may also be changed, as well as the basis
for the preamble (`preex`).

"""
function _populate!(ex::Expr, preex::Expr, d_indx, v_vars, v_assign, v_preamb)

    indx_rm = Int[]   # Book-keeping: assignements to be deleted
    for (i, aa) in enumerate(ex.args)
        if (aa.head == :for) || (aa.head == :block)
            push!(preex.args, Expr(aa.head))
            (aa.head == :for) && push!(preex.args[end].args, aa.args[1])
            _populate!(aa, preex.args[end], d_indx, v_vars, v_assign, v_preamb)
            #
        else
            # This assumes aa.head == :(:=)
            aa_lhs = aa.args[1]
            aa_rhs = aa.args[2]

            if isa(aa_rhs, Expr)
                (aa_rhs.args[1] == :eachindex || aa_rhs.head == :(:)) && continue

                fnexpr, def_fnexpr, auxfnexpr =
                    _replacecalls!(aa_rhs, aa_lhs, d_indx, v_vars)
                !in(def_fnexpr.args[1], keys(v_preamb)) &&
                    push!(v_preamb, def_fnexpr.args[1] => def_fnexpr.args[2])
                push!(preex.args, def_fnexpr)

                ex.args[i] = fnexpr
                if auxfnexpr.head != :nothing &&
                        !in(auxfnexpr.args[1], keys(v_preamb))
                    push!(v_preamb, auxfnexpr.args[1] => auxfnexpr.args[2])
                    push!(preex.args, auxfnexpr)
                end

                !in(aa_lhs, v_vars) && push!(v_vars, subs(aa_lhs, d_indx))
                #
            elseif isa(aa_rhs, Symbol) # occurs when there is a simple assignment
                bb = subs(aa, Dict(aa_rhs => :(identity($aa_rhs))))
                bb_lhs = bb.args[1]
                bb_rhs = bb.args[2]

                fnexpr, def_fnexpr, auxfnexpr =
                    _replacecalls!(bb_rhs, bb_lhs, d_indx, v_vars)
                nvar = def_fnexpr.args[1]
                !in(nvar, keys(v_preamb)) &&
                    push!(v_preamb, nvar => def_fnexpr.args[2])
                push!(preex.args, def_fnexpr)

                ex.args[i] = fnexpr
                if auxfnexpr.head != :nothing &&
                        !in(auxfnexpr.args[1], keys(v_preamb))
                    push!(v_preamb, auxfnexpr.args[1] => auxfnexpr.args[2])
                    push!(preex.args, auxfnexpr)
                end

                # !in(bb_lhs, v_vars) && push!(v_vars, bb_lhs)
                !in(bb_lhs, v_vars) && push!(v_vars, subs(bb_lhs, d_indx))
                #
            elseif isa(aa_rhs, Number)
                push!(v_assign, aa_lhs => aa_rhs)
                push!(indx_rm, i)
                #
            else #needed?
                @show(aa.args[2], typeof(aa.args[2]))
                error("Different from `Expr`, `Symbol` or `Number`")
                #
            end
        end
    end

    # Delete assignement statements
    length(indx_rm) !=0 && deleteat!(ex.args, indx_rm)

    return nothing
end


"""
`_rename_indexedvars(fnbody)`

Renames the indexed variables (using `Espresso.genname()`) that
exists in `fnbody`. Returns `fnbody` with the renamed variables
and a dictionary that links the new variables to the old
indexed ones.

"""
# Thanks to Andrei Zhabinski (@dfdx, see #31) for this implementation
# now slightly modified
function _rename_indexedvars(fnbody)
    indexed_vars = findex(:(_X[_i...]), fnbody)
    st = Dict(ivar => genname() for ivar in indexed_vars)
    new_fnbody = subs(fnbody, st)
    return new_fnbody, Dict(v => k for (k, v) in st)
end


"""
`_preamble_body(fnbody, fnargs, debug=false)`

Returns the preamble, the body and a guessed return
variable, which will be used to build the parsed
function. `fnbody` is the expression with the body of
the original function, `fnargs` is a vector of symbols
of the original diferential equations function.

"""
function _preamble_body(fnbody, fnargs, debug=false)

    # Bookkeeping
    v_vars = Union{Symbol,Expr}[]                  # List of symbols with created variables
    v_preamb = Dict{Union{Symbol,Expr},Expr}()     # Auxiliary definitions
    v_assign = Pair{Union{Symbol,Expr}, Number}[]  # Numeric assignments (to be deleted)

    # Rename vars to have the body in non-indexed form
    fnbody, d_indx = _rename_indexedvars(fnbody)
    debug && (@show(d_indx); println())

    # Create newfnbody
    newfnbody = _newfnbody(fnbody)
    # Needed, if `newfnbody` consists of a single assignment (unary call)
    if newfnbody.head == :(=)
        newfnbody = Expr(:block, newfnbody)
    end
    debug && (@show(newfnbody); println())

    # Populate v_vars, v_assign]
    preamble = Expr(:block,)
    _populate!(newfnbody, preamble, d_indx, v_vars, v_assign, v_preamb)
    preamble = preamble.args[1]
    debug && (@show(v_vars, v_assign, v_preamb, d_indx); println())
    debug && (@show(preamble); println())
    debug && (@show(newfnbody); println())

    # Substitute numeric assignments and indexed variables in new function's body
    newfnbody = subs(newfnbody, Dict(v_assign))
    preamble = subs(preamble, Dict(v_assign))

    # Include the assignement of indexed auxiliary variables
    defspreamble = _indexed_definitions!(preamble::Expr, d_indx, fnargs)
    debug && (@show(d_indx); println())
    # Update substitutions
    preamble = subs(preamble, d_indx)
    newfnbody = subs(newfnbody, d_indx)
    prepend!(preamble.args, defspreamble)
    debug && (@show(preamble); println())
    debug && (@show(newfnbody); println())

    # Define retvar; for scalar eqs is the last included in v_vars
    retvar = length(fnargs) == 2 ? subs(v_vars[end], d_indx) : fnargs[end]

    return preamble, newfnbody, retvar
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
    forloopblock = Expr(:for, :(ord = 1:order-1),
        Expr(:block, :(ordnext = ord + 1)) )

    # Transform graph representation of the body of the function
    preamble, fnbody, retvar = _preamble_body(fnbody, fnargs, debug)
    debug && (@show(preamble, fnbody, retvar); println())

    # Recursion relation
    if length(fnargs) == 2
        rec_preamb = :( $(fnargs[2])[2] = $(retvar)[1] )
        rec_fnbody = :( $(fnargs[2])[ordnext+1] =
            $(retvar)[ordnext]/ordnext )
    elseif length(fnargs) == 3
        retvar = fnargs[end]
        rec_preamb = :(
            @inbounds for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[2] = $(retvar)[__idx].coeffs[1]
            end
        )
        deleteat!(rec_preamb.args[2].args[2].args, 1)
        rec_fnbody = :(
            @inbounds for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[ordnext+1] =
                    $(retvar)[__idx].coeffs[ordnext]/ordnext
            end
        )
        deleteat!(rec_fnbody.args[2].args[2].args, 1)
    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    # Add preamble to newfunction
    push!(newfunction.args[2].args, preamble.args..., rec_preamb)

    # Add parsed fnbody to forloopblock
    push!(forloopblock.args[2].args, fnbody.args[1].args..., rec_fnbody)

    # Push preamble and forloopblock to newfunction
    push!(newfunction.args[2].args, forloopblock);
    push!(newfunction.args[2].args, parse("return nothing"))

    # Rename variables of the body of the new function
    newfunction = subs(newfunction,
        Dict(fnargs[1] => :(__tT), fnargs[2] => :(__x), :(__fn) => fn ))
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(fnargs[3] => :(__dx)))
    end

    newfunction
end


"""
`@taylorize_ode ex`

Used only when `ex` is the definition of a function. It
evaluates `ex` and also the parsed function corresponding
to `ex` in terms of the mutating functions of TaylorSeries.

"""
macro taylorize_ode( ex )
    nex = _make_parsed_jetcoeffs(ex)
    # @show(ex)
    println()
    # @show(nex)
    quote
        eval( $(esc(ex)) )  # evals to calling scope the passed function
        eval( $(esc(nex)) ) # New method of `jetcoeffs!`
    end
end
