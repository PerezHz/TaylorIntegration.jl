# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of MacroTools and Espresso
using MacroTools: @capture, shortdef

using Espresso: subs, simplify, ExGraph, to_expr, sanitize, genname,
    find_vars, findex, find_indices, isindexed


# Define some constants to create the new (parsed) functions
# The (irrelevant) `nothing` below is there to have a :block Expr; deleted later
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function jetcoeffs!(__tT::Taylor1{T}, __x::Taylor1{S}, ::Type{Val{__fn}}) where
        {T<:Real, S<:Number}

    order = __tT.order
    nothing
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function jetcoeffs!(__tT::Taylor1{T}, __x::Vector{Taylor1{S}},
        __dx::Vector{Taylor1{S}}, ::Type{Val{__fn}}) where {T<:Real, S<:Number}

    order = __tT.order
    nothing
end)
);

# Constants for the initial declaration and initialization of arrays
const _DECL_ARRAY = Expr(:block,
    :(__var1 = Array{Taylor1{S}}(__var2)),
    :(@__dot__  __var1 = Taylor1( zero(S), order )) )


"""
`_make_parsed_jetcoeffs( ex, debug=false )`

This function constructs a new function, equivalent to the
differential equations, which exploits the mutating functions
of TaylorSeries.jl.

"""
function _make_parsed_jetcoeffs( ex, debug=false )

    # Extract the name, args and body of the function
    fn, fnargs, fnbody = _extract_parts(ex)
    debug && (println("****** _extract_parts ******");
        @show(fn, fnargs, fnbody); println())

    # Set up new function
    newfunction = _newhead(fn, fnargs)

    # for-loop block (recursion); it will contain the parsed body
    forloopblock = Expr(:for, :(ord = 1:order-1),
        Expr(:block, :(ordnext = ord + 1)) )

    # Transform graph representation of the body of the function
    defspreamble, fnbody, retvar = _preamble_body(fnbody, fnargs, debug)
    debug && (println("****** _preamble_body ******");
        @show(defspreamble); println(); @show(fnbody); println();)

    # Recursion relation
    debug && println("****** _recursionloop ******")
    rec_preamb, rec_fnbody = _recursionloop(fnargs, retvar)

    # Add preamble to newfunction
    push!(newfunction.args[2].args, defspreamble..., rec_preamb)

    # Add parsed fnbody to forloopblock
    push!(forloopblock.args[2].args, fnbody.args[1].args..., rec_fnbody)

    # Push preamble and forloopblock to newfunction
    push!(newfunction.args[2].args, forloopblock, parse("return nothing"))

    # Rename variables of the body of the new function
    newfunction = subs(newfunction, Dict(:__tT => fnargs[1],
        :(__x) => fnargs[2], :(__fn) => fn ))
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(:(__dx) => fnargs[3]))
    end

    return newfunction
end



"""
`_extract_parts(ex::Expr)`

Returns the function name, the function arguments, and the body of
a function passed as an `Expr`. The function may be provided as
a one-line function, or in the long form (anonymous functions
do not work).

"""
function _extract_parts(ex)

    # Capture name, args and body
    @capture( shortdef(ex), ffn_(ffnargs__) = fnbody_ ) ||
        throw(ArgumentError("It must be a function call:\n $ex"))

    # Clean-up `ffn`
    fn = isa(ffn, Symbol) ? ffn : ffn.args[1]

    # Clean-up (from declarations) `ffnargs`
    fnargs = []
    for ff in ffnargs
        if isa(ff, Symbol)
            push!(fnargs, ff)
        elseif isa(ff, Expr)
            push!(fnargs, ff.args[1])
        else
            throw(ArgumentError("$ff is neither a `Symbol` nor an `Expr`"))
        end
    end

    # Standarize fnbody, same structure for one-line or long-form functions
    if length(fnbody.args) > 1
        fnbody.args[1] = Expr(:block, copy(fnbody.args)...)
        deleteat!(fnbody.args, 2:length(fnbody.args))
    end

    # Special case: the last arg of the function is a simple
    # assignement (symbol), a numerical value or simply `nothing`
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

    # Delete the irrelevant `nothing`
    pop!(newfunction.args[2].args)

    return newfunction
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

    # Bookkeeping:
    #   v_vars:   List of symbols with created variables
    #   v_assign: Numeric assignments (to be deleted)
    v_vars = Union{Symbol,Expr}[]
    v_assign = Dict{Union{Symbol,Expr}, Number}()

    # Rename vars to have the body in non-indexed form
    fnbody, d_indx = _rename_indexedvars(fnbody)
    debug && (println("------ _rename_indexedvars ------");
        @show(fnbody); println(); @show(d_indx); println())

    # Create newfnbody
    #    v_newindx: symbols of auxiliary indexed vars
    #    v_arraydecl: symbols which are explicitly declared as Array or Vector
    newfnbody, v_newindx, v_arraydecl = _newfnbody(fnbody, d_indx)
    debug && (println("------ _newfnbody ------");
        @show(v_newindx); println(); @show(v_arraydecl); println();
        @show(newfnbody); println())

    # Parse `newfnbody!` and create `prepreamble`, updating the
    # bookkeeping vectors.
    prepreamble = Expr(:block,)
    _parse_newfnbody!(newfnbody, prepreamble, v_vars, v_assign, d_indx,
        v_newindx, v_arraydecl)
    preamble = prepreamble.args[1]

    # Substitute numeric assignments in new function's body
    newfnbody = subs(newfnbody, Dict(v_assign))
    preamble = subs(preamble, Dict(v_assign))
    debug && (println("------ _parse_newfnbody! ------");
        @show(newfnbody); println(); @show(preamble); println();
        @show(v_vars); println(); @show(d_indx); println();
        @show(v_newindx); println())

    # Include the assignement of indexed auxiliary variables
    defspreamble = _defs_preamble!(preamble, fnargs,
        d_indx, v_newindx, v_arraydecl, Union{Symbol,Expr}[], similar(d_indx))
    # preamble = subs(preamble, d_indx)

    # Bring back substitutions
    newfnbody = subs(newfnbody, d_indx)

    # Define retvar; for scalar eqs is the last included in v_vars
    retvar = length(fnargs) == 2 ? subs(v_vars[end], d_indx) : fnargs[end]

    debug && (println("------ _defs_preamble! ------");
        @show(d_indx); println(); @show(v_newindx); println();
        @show(newfnbody); println(); @show(retvar); println())

    return defspreamble, newfnbody, retvar
end



"""
`_rename_indexedvars(fnbody)`

Renames the indexed variables (using `Espresso.genname()`) that
exists in `fnbody`. Returns `fnbody` with the renamed variables
and a dictionary that links the new variables to the old
indexed ones.

"""
function _rename_indexedvars(fnbody)
    # Thanks to Andrei Zhabinski (@dfdx, see #31) for this
    # implementation, now slightly modified
    indexed_vars = findex(:(_X[_i...]), fnbody)
    st = Dict(ivar => genname() for ivar in indexed_vars)
    new_fnbody = subs(fnbody, st)
    return new_fnbody, Dict(v => k for (k, v) in st)
end



"""
`_newfnbody(fnbody, d_indx)`

Returns a new (modified) body of the function, a priori unfolding
the expression graph (AST) as unary and binary calls, and a vector
(`v_newindx`) with the name of auxiliary indexed variables.

"""
function _newfnbody(fnbody, d_indx)
    # `fnbody` is assumed to be a `:block` `Expr`
    newfnbody = Expr(:block,)
    v_newindx = Symbol[]
    v_arraydecl = Symbol[]

    for (i, ex) in enumerate(fnbody.args)
        if isa(ex, Expr)

            # Ignore the following cases
            (ex.head == :line || ex.head == :return) && continue

            # Treat `for` loops separately
            if ex.head == :block
                newblock, tmp_newindx, tmp_arraydecl = _newfnbody(ex, d_indx)
                push!(newfnbody.args, newblock )
                append!(v_newindx, tmp_newindx)
                append!(v_arraydecl, tmp_arraydecl)
            elseif ex.head == :for
                push!(newfnbody.args, Expr(:for, ex.args[1]))
                loopbody, tmp_newindx, tmp_arraydecl =
                    _newfnbody( ex.args[2], d_indx )
                push!(newfnbody.args[end].args, loopbody)
                append!(v_newindx, tmp_newindx)
                append!(v_arraydecl, tmp_arraydecl)
            elseif ex.head == :if
                push!(newfnbody.args, Expr(:if, ex.args[1]))
                for exx in ex.args[2:end]
                    ifbody, tmp_newindx, tmp_arraydecl = _newfnbody( exx, d_indx)
                    push!(newfnbody.args[end].args, ifbody)
                    append!(v_newindx, tmp_newindx)
                    append!(v_arraydecl, tmp_arraydecl)
                end
            elseif ex.head == :(=) || ex.head == :call

                ex_lhs = ex.args[1]
                ex_rhs = ex.args[2]

                # Case of explicit declaration of Array or Vector
                if !isempty(findex(:(_Array{_TT...}), ex)) ||
                        !isempty(findex(:(Vector{_TT...}), ex))
                    push!(v_arraydecl, ex_lhs)
                    push!(newfnbody.args, ex)
                    continue
                end

                # Unfold AST graph
                nex = deepcopy(ex)
                try
                    nex = to_expr(ExGraph(simplify(ex)))
                catch
                    push!(newfnbody.args, ex)
                    continue
                end
                # @show(ex, nex)
                push!(newfnbody.args, nex.args[2:end]...)

                # Bookkeeping of indexed vars, to define assignements
                isindx_lhs = haskey(d_indx, ex_lhs)
                for nexargs in nex.args[2:end]
                    # (nexargs.head == :line ||
                    #     haskey(d_indx, nexargs.args[1])) && continue
                    haskey(d_indx, nexargs.args[1]) && continue
                    vars_nex = find_vars(nexargs)

                    # any(haskey.(d_indx, vars_nex[:])) &&
                    #     !in(vars_nex[1], v_newindx) &&
                    (isindx_lhs || vars_nex[1] != ex_lhs) &&
                        push!(v_newindx, vars_nex[1])
                end
            else
                @show(typeof(ex), ex.head)
                throw(ArgumentError("$ex.head is not yet implemented"))
            end
            #
        else
            @show(typeof(ex))
            throw(ArgumentError("$ex is not an `Expr`"))
            #
        end
    end

    # Needed, if `newfnbody` consists of a single assignment (unary call)
    if newfnbody.head == :(=)
        newfnbody = Expr(:block, newfnbody)
    end

    return newfnbody, v_newindx, v_arraydecl
end



"""
`_parse_newfnbody!(ex, preex, v_vars, v_assign, d_indx, v_newindx, v_arraydecl,
    [inloop=false])`

Parses `ex` (the new body of the function) replacing the expressions
to use the mutating functions of TaylorSeries, and building the preamble
`preex`. This is done by traversing recursively the args of `ex`, updating
the bookkeeping vectors `v_vars` and `v_assign`. `d_indx` is
used to substitute back the explicit indexed variables.

"""
function _parse_newfnbody!(ex::Expr, preex::Expr,
        v_vars, v_assign, d_indx, v_newindx, v_arraydecl, inloop=false)

    # Numeric assignements to be deleted
    indx_rm = Int[]

    for (i, aa) in enumerate(ex.args)

        # Treat for loops and blocks separately
        if (aa.head == :for)
            push!(preex.args, Expr(:for, aa.args[1]))
            _parse_newfnbody!(aa, preex.args[end],
                v_vars, v_assign, d_indx, v_newindx, v_arraydecl, true)
            #
        elseif (aa.head == :block)

            push!(preex.args, Expr(:block))
            _parse_newfnbody!(aa, preex.args[end],
                v_vars, v_assign, d_indx, v_newindx, v_arraydecl)
            #
        elseif (aa.head == :if)
            push!(preex.args, Expr(:if, aa.args[1]))
            for exx in aa.args[2:end]
                push!(preex.args[end].args, Expr(:block))
                _parse_newfnbody!(exx, preex.args[end].args[end],
                    v_vars, v_assign, d_indx, v_newindx, v_arraydecl)
            end
            #
        elseif (aa.head == :(=))

            i == 1 && inloop && continue

            # Main case
            aa_lhs = aa.args[1]
            aa_rhs = aa.args[2]

            # Replace expressions when needed, and bookkeeping
            if isa(aa_rhs, Expr)
                (aa_rhs.args[1] == :eachindex || aa_rhs.head == :(:)) && continue
                _replace_expr!(ex, preex, i,
                    aa_lhs, aa_rhs, v_vars, d_indx, v_newindx)

                # Remove new aa_lhs declaration from `ex`; it is already declared
                in(aa_lhs, v_arraydecl) && push!(indx_rm, i)
                #
            elseif isa(aa_rhs, Symbol) # occurs when there is a simple assignment

                bb = subs(aa, Dict(aa_rhs => :(identity($aa_rhs))))
                bb_lhs = bb.args[1]
                bb_rhs = bb.args[2]
                _replace_expr!(ex, preex, i,
                    bb_lhs, bb_rhs, v_vars, d_indx, v_newindx)
                #
            elseif isa(aa_rhs, Number)
                push!(v_assign, aa_lhs => aa_rhs)
                push!(indx_rm, i)
                #
            else #needed?
                @show(aa, typeof(aa_rhs[2]))
                error("Different from `Expr`, `Symbol` or `Number`")
                #
            end
            #
        else

            # Unrecognized head; pass the expression as it is
            push!(preex.args, aa)
        end
    end

    # Delete trivial numeric assignement statements
    isempty(indx_rm) || deleteat!(ex.args, indx_rm)

    return nothing
end



"""
`_replace_expr!(ex, preex, i, aalhs, aarhs, v_vars, d_indx, v_newindx)`

Replaces the calls in `ex.args[i]`, and updates `preex` with the definitions
of the expressions, based on the the LHS (`aalhs`) and RHS (`aarhs`) of the
base assignment. The bookkeeping vectors (`v_vars`, `v_newindx`)
are updated. `d_indx` is used to bring back the indexed variables.

"""
function _replace_expr!(ex::Expr, preex::Expr, i::Int,
        aalhs, aarhs, v_vars, d_indx, v_newindx)

    # Replace calls
    fnexpr, def_fnexpr, auxfnexpr = _replacecalls!(aarhs, aalhs, v_vars)

    # Bring back indexed variables
    fnexpr = subs(fnexpr, d_indx)
    def_fnexpr = subs(def_fnexpr, d_indx)
    auxfnexpr = subs(auxfnexpr, d_indx)

    # Update `ex` and `preex`
    push!(preex.args, def_fnexpr)
    ex.args[i] = fnexpr

    # Same for the auxiliary expressions
    if auxfnexpr.head != :nothing
        push!(preex.args, auxfnexpr)
        # in(aalhs, v_newindx) && push!(v_newindx, auxfnexpr.args[1])
        push!(v_newindx, auxfnexpr.args[1])
    end

    !in(aalhs, v_vars) && push!(v_vars, subs(aalhs, d_indx))
    return nothing
end



"""
`_replacecalls!(fnold, newvar, v_vars)`

Replaces the symbols of unary and binary calls
of the expression `fnold`, which defines `newvar`,
by the mutating functions in TaylorSeries.jl.
The vector `v_vars` is updated if new auxiliary
variables are introduced.

"""
function _replacecalls!(fnold::Expr, newvar::Symbol, v_vars)

    ll = length(fnold.args)
    dcall = fnold.args[1]
    newarg1 = fnold.args[2]

    # If call is not in mutating functions dictionaries, copy original one
    if !( in(dcall, keys(TaylorSeries._dict_unary_calls)) ||
            in(dcall, keys(TaylorSeries._dict_binary_calls)) )

        fnexpr = :($newvar = $fnold)
        def_fnexpr = fnexpr
        aux_fnexpr = Expr(:nothing)
        return fnexpr, def_fnexpr, aux_fnexpr
    end

    if ll == 2

        # Unary call
        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_unary_calls[dcall]
        fnexpr = subs(fnexpr,
            Dict(:_res => newvar, :_arg1 => newarg1, :_k => :ord))
        def_fnexpr = :( _res = Taylor1($(def_fnexpr.args[2]), order) )
        def_fnexpr = subs(def_fnexpr,
            Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))),
                :_k => :ord))

        # Auxiliary expression
        if aux_fnexpr.head != :nothing
            newaux = genname()
            aux_fnexpr = :( _res = Taylor1($(aux_fnexpr.args[2]), order) )
            aux_fnexpr = subs(aux_fnexpr,
                Dict(:_res => newaux, :_arg1 => :(constant_term($(newarg1))),
                    :_aux => newaux))
            fnexpr = subs(fnexpr, Dict(:_aux => newaux))
            !in(newaux, v_vars) && push!(v_vars, newaux)
        end
    elseif ll == 3

        # Binary call; no auxiliary expressions needed
        newarg2 = fnold.args[3]

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_binary_calls[dcall]
        fnexpr = subs(fnexpr,
            Dict(:_res => newvar, :_arg1 => newarg1, :_arg2 => newarg2,
                :_k => :ord))

        def_fnexpr = :(_res = Taylor1($(def_fnexpr.args[2]), order) )
        def_fnexpr = subs(def_fnexpr,
            Dict(:_res => newvar,
                :_arg1 => :(constant_term($(newarg1))),
                :_arg2 => :(constant_term($(newarg2))),
                :_k => :ord))
    else
        # Recognized call, but not as a unary or binary call
        fnexpr = :($newvar = $fnold)
        def_fnexpr = fnexpr
        aux_fnexpr = Expr(:nothing)
    end

    return fnexpr, def_fnexpr, aux_fnexpr
end



"""
`_defs_preamble!(preamble, fnargs, d_indx, v_newindx, v_arraydecl, v_preamb,
    d_decl, [inloop=false])`

Returns a vector with expressions defining the auxiliary variables
in the preamble; it may modify `d_indx` if new variables are introduced.
`v_preamb` is for bookkeeping the introduced variables

"""
function _defs_preamble!(preamble::Expr, fnargs,
        d_indx, v_newindx, v_arraydecl, v_preamb, d_decl, inloop::Bool=false,
        ex_aux::Expr = Expr(:block,))

    # Initializations
    defspreamble = Expr[]

    for (i, ex) in enumerate(preamble.args)

        if isa(ex, Expr)

            # Treat :block and :for separately
            if (ex.head == :block)
                newdefspr = _defs_preamble!(ex, fnargs, d_indx,
                    v_newindx, v_arraydecl, v_preamb, d_decl, inloop, ex_aux)
                append!(defspreamble, newdefspr)
            elseif (ex.head == :for)
                push!(ex_aux.args, ex.args[1])
                newdefspr = _defs_preamble!(ex.args[2], fnargs, d_indx,
                    v_newindx, v_arraydecl, v_preamb, d_decl, true, ex_aux)
                append!(defspreamble, newdefspr)
                pop!(ex_aux.args)
            elseif (ex.head == :if)
                for exx in ex.args[2:end]
                    newdefspr = _defs_preamble!(exx, fnargs, d_indx,
                        v_newindx, v_arraydecl, v_preamb, d_decl, inloop, ex_aux)
                    append!(defspreamble, newdefspr)
                end
                continue
            elseif (ex.head == :(=))

                # `ex.head` is a :(=) of some kind
                alhs = ex.args[1]
                arhs = subs(ex.args[2], d_indx) # substitute updated vars in rhs

                # Outside of a loop
                if !inloop
                    in(alhs, v_preamb) && continue
                    push!(defspreamble, subs(ex, d_decl))
                    push!(v_preamb, alhs)
                    continue
                end

                # Inside a loop
                if isindexed(alhs)

                    # `var1` may be a vector or a matrix; declaring it is subtle
                    var1 = alhs.args[1]
                    (in(var1, fnargs) || in(var1, v_arraydecl) ||
                        in(alhs, v_preamb)) && continue

                    # indices of var1
                    d_subs = Dict(veexx.args[1] => veexx.args[2]
                        for veexx in ex_aux.args)
                    indx1 = alhs.args[2:end]
                    ex_tuple = :( [$(indx1...)] )
                    ex_tuple = subs(ex_tuple, d_subs)
                    ex_tuple = :( size( $(Expr(:tuple, ex_tuple.args...)) ))
                    exx = subs(_DECL_ARRAY, Dict(:__var1 => :($var1),
                        :__var2 => :($ex_tuple)) )
                    push!(defspreamble, exx.args...)
                    push!(v_preamb, var1)
                    continue
                    #
                elseif in(alhs, v_newindx) || isindexed(arhs)
                    # `alhs` is an aux indexed var, so something in `arhs`
                    # is indexed.

                    in(alhs, v_preamb) && continue

                    vars_indexed = findex(:(_X[_i...]), arhs)

                    # NOTE: Use the size of the var with more indices
                    # to define the declaration of the new array.
                    iimax, ii_indx = findmax(
                        [length(find_indices(aa)[1]) for aa in vars_indexed] )
                    var1 = vars_indexed[ii_indx].args[1]
                    indx1 = vars_indexed[ii_indx].args[2:end]

                    exx_indx = ones(Int, length(indx1))
                    push!(d_indx, alhs => :($alhs[$(indx1...)]) )
                    push!(d_decl, alhs => :($alhs[$exx_indx...]))
                    exx = subs(_DECL_ARRAY,
                        Dict(:__var1 => :($alhs), :__var2 => :(size($var1))) )
                    push!(defspreamble, exx.args...)
                    push!(v_preamb, alhs)
                    continue
                    #
                else

                    # `alhs` is not indexed nor an aux indexed var
                    in(alhs, v_preamb) && continue
                    vars_indexed = findex(:(_X[_i...]), arhs)
                    if !isempty(vars_indexed)
                        ex = subs(ex, Dict(vv => :(one(S)) for vv in vars_indexed))
                    end
                    ex = subs(ex, d_decl)
                    push!(defspreamble, ex)
                    push!(v_preamb, alhs)
                    continue
                    #
                end
                #
            end
        else
            @show(i, ex, typeof(ex))
            throw(ArgumentError("$ex is not an `Expr`"))
            #
        end

        exx = subs(ex, d_indx)
        !inloop && push!(defspreamble, exx)

    end
    return defspreamble
end



"""
`_recursionloop(fnargs, retvar)`

Build the expression for the recursion-loop.

"""
function _recursionloop(fnargs, retvar)

    ll = length(fnargs)
    if ll == 2

        rec_preamb = sanitize( :( $(fnargs[2])[1] = $(retvar)[0] ) )
        rec_fnbody = sanitize(
            :( $(fnargs[2])[ordnext] = $(retvar)[ord]/ordnext ) )
        #
    elseif ll == 3

        retvar = fnargs[end]
        rec_preamb = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                @inbounds $(fnargs[2])[__idx].coeffs[2] =
                    $(retvar)[__idx].coeffs[1]
            end
        ))
        rec_fnbody = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                @inbounds $(fnargs[2])[__idx].coeffs[ordnext+1] =
                    $(retvar)[__idx].coeffs[ordnext]/ordnext
            end
        ))
        #
    else

        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end
    return rec_preamb, rec_fnbody
end



"""
`@taylorize ex`

Used only when `ex` is the definition of a function. It
evaluates `ex` and also the parsed function corresponding
to `ex` in terms of the mutating functions of TaylorSeries.

"""
macro taylorize( ex )
    nex = _make_parsed_jetcoeffs(ex)
    # @show(ex)
    println()
    # @show(nex)
    quote
        eval( $(esc(ex)) )  # evals to calling scope the passed function
        eval( $(esc(nex)) ) # New method of `jetcoeffs!`
    end
end
