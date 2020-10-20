# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of Espresso
using Espresso: subs, simplify, ExGraph, ExH, to_expr, sanitize, genname,
    find_vars, findex, find_indices, isindexed


# Define some constants to create the new (parsed) functions
# The (irrelevant) `nothing` below is there to have a :block Expr; deleted later
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
function TaylorIntegration.jetcoeffs!(::Val{__fn}, __tT::Taylor1{_T}, __x::Taylor1{_S}, __params) where
        {_T<:Real, _S<:Number}

    order = __tT.order
    nothing
end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
function TaylorIntegration.jetcoeffs!( ::Val{__fn}, __tT::Taylor1{_T}, __x::AbstractArray{Taylor1{_S}, _N},
        __dx::AbstractArray{Taylor1{_S}, _N}, __params) where {_T<:Real, _S<:Number, _N}

    order = __tT.order
    nothing
end)
);

# Constants for the initial declaration and initialization of arrays
const _DECL_ARRAY = sanitize( Expr(:block,
    :(__var1 = Array{Taylor1{_S}}(undef, __var2)),
    :(__var1 .= Taylor1( zero(_S), order ))
    ) )


"""
`_make_parsed_jetcoeffs( ex, debug=false )`

This function constructs a new function, equivalent to the
differential equations, which exploits the mutating functions
of TaylorSeries.jl.

"""
function _make_parsed_jetcoeffs(ex::Expr, debug=false)

    # Extract the name, args and body of the function
    fn, fnargs, fnbody = _extract_parts(ex)
    debug && (println("****** _extract_parts ******");
        @show(fn, fnargs, fnbody); println())

    # Set up new function
    newfunction = _newhead(fn, fnargs)

    # for-loop block (recursion); it will contain the parsed body
    forloopblock = Expr(:for, :(ord = 1:order-1),
        Expr(:block, :(ordnext = ord + 1)) )

    #= Transform graph representation of the body of the function
    - `defspreamble` includes the definitions used for zeroth order (preamble)
    - `fnbody` is the transformed function body, using mutating functions
        from TaylorSeries, and is used within the for-loop block
    - `retvar` is the (guessed) return variable, which defines the LHS
        of the ODEs
    =#
    defspreamble, fnbody, retvar = _preamble_body(fnbody, fnargs, debug)
    debug && (println("****** _preamble_body ******");
        @show(defspreamble); println(); @show(fnbody); println();)

    #= Create body of recursion loop; temporary assignements may be needed.
    - rec_preamb
    - rec_fnbody
    =#
    debug && println("****** _recursionloop ******")
    rec_preamb, rec_fnbody = _recursionloop(fnargs, retvar)

    # Add preamble to newfunction
    push!(newfunction.args[2].args, defspreamble..., rec_preamb)

    # Add parsed fnbody to forloopblock
    push!(forloopblock.args[2].args, fnbody.args[1].args..., rec_fnbody)

    # Push preamble and forloopblock to newfunction
    push!(newfunction.args[2].args, forloopblock, Meta.parse("return nothing"))

    # Rename variables of the body of the new function
    if length(fnargs) == 3
        newfunction = subs(newfunction, Dict(:__tT => fnargs[3],
            :__params => fnargs[2], :(__x) => fnargs[1], :(__fn) => fn ))
    elseif length(fnargs) == 4
        newfunction = subs(newfunction, Dict(:__tT => fnargs[4],
            :__params => fnargs[3], :(__x) => fnargs[2], :(__dx) => fnargs[1],
            :(__fn) => fn ))
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
function _extract_parts(ex::Expr)

    # Capture name, args and body
    # @capture( shortdef(ex), ffn_(ffnargs__) = fnbody_ ) ||
    #     throw(ArgumentError("It must be a function call:\n $ex"))
    dd = Dict{Symbol, Any}()
    _capture_fn_args_body!(ex, dd)
    ffn, ffnargs, fnbody = dd[:fname], dd[:fnargs], dd[:fnbody]

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
            fnbody.args[1].args[end] = :( identity($(fnbody.args[1].args[end])) )
        end
    elseif isa(fnbody.args[1].args[end], Number)
        fnbody.args[1].args[end] = :( $(fnbody.args[1].args[end])+zero($(fnargs[1])) )
    end

    return fn, fnargs, fnbody
end



"""
`_capture_fn_args_body!(ex, vout::Dict{Symbol, Any})`

Captures the name of a function, arguments, body and other properties,
returning them as the values of the dictionary `dd`, which is updated
in place.

"""
function _capture_fn_args_body!(ex::Expr, vout::Dict{Symbol, Any} = Dict())
    exh = ExH(ex)
    if exh.head == :(=) || exh.head == :function
        _capture_fn_args_body!.(ex.args, Ref(vout))
    elseif exh.head == :call
        push!(vout, Pair(:fname, exh.args[1]))
        push!(vout, Pair(:fnargs, exh.args[2:end]))
    elseif exh.head == :block
        push!(vout, Pair(:fnbody, ex))
    elseif exh.head == :where
        push!(vout, Pair(:where, ex.args[2:end]))
        _capture_fn_args_body!(ex.args[1], vout)
    end
    vout
end



"""
`_newhead(fn, fnargs)`

Creates the head of the new method of `jetcoeffs!`. Here,
`fn` is the name of the passed function and `fnargs` is a
vector with its arguments (which are two or three).

"""
function _newhead(fn, fnargs)

    # Construct common elements of the new expression
    if length(fnargs) == 3
        newfunction = copy(_HEAD_PARSEDFN_SCALAR)
    elseif length(fnargs) == 4
        newfunction = copy(_HEAD_PARSEDFN_VECTOR)
    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

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

    #= Rename vars to have the body in non-indexed form
    `d_indx` is a dictionary mapping new variables (symbols) to old
    (perhaps indexed) symbols
    =#
    fnbody, d_indx = _rename_indexedvars(fnbody)
    debug && (println("------ _rename_indexedvars ------");
        @show(fnbody); println(); @show(d_indx); println())

    #= Create newfnbody
    - v_newindx: symbols of auxiliary indexed vars
    - v_arraydecl: symbols which are explicitly declared as Array or Vector
    - `newfnbody` corresponds to `fnbody`, cleaned (without irrelevant comments)
        and with all new variables in place
    =#
    newfnbody, v_newindx, v_arraydecl = _newfnbody(fnbody, fnargs, d_indx)
    debug && (println("------ _newfnbody ------");
        @show(v_newindx); println(); @show(v_arraydecl); println();
        @show(newfnbody); println())

    # Parse `newfnbody` and create `prepreamble`, updating the bookkeeping vectors.
    # The returned `newfnbody` and `prepreamble` use the mutating functions
    # of TaylorSeries.
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
        @show(v_assign); println(); @show(v_newindx); println())

    # Include the assignement of indexed auxiliary variables
    defspreamble = _defs_preamble!(preamble, fnargs,
        d_indx, v_newindx, v_arraydecl, Union{Symbol,Expr}[], empty(d_indx))
    # preamble = subs(preamble, d_indx)

    # Bring back substitutions
    newfnbody = subs(newfnbody, d_indx)

    # Define retvar; for scalar eqs is the last entry included in v_vars
    retvar = length(fnargs) == 3 ? subs(v_vars[end], d_indx) : fnargs[1]

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
`_newfnbody(fnbody, fnargs, d_indx)`

Returns a new (modified) body of the function, a priori unfolding
the expression graph (AST) as unary and binary calls, and a vector
(`v_newindx`) with the name of auxiliary indexed variables.

"""
function _newfnbody(fnbody, fnargs, d_indx)
    # `fnbody` is assumed to be a `:block` `Expr`
    newfnbody = Expr(:block,)
    v_newindx = Symbol[]
    v_arraydecl = Symbol[]

    # The magic happens HERE!!
    # Each line of fnbody (fnbody.args) is parsed separately
    for (i, ex) in enumerate(fnbody.args)
        if isa(ex, Expr)
            ex_head = ex.head

            # Ignore the following cases
            (ex_head == :return) && continue

            # Treat `for` loops, `Threads.@threads for` and `if` blocks separately
            if ex_head == :block
                newblock, tmp_newindx, tmp_arraydecl = _newfnbody(ex, fnargs, d_indx)
                push!(newfnbody.args, newblock )
                append!(v_newindx, tmp_newindx)
                append!(v_arraydecl, tmp_arraydecl)
            elseif ex_head == :for
                push!(newfnbody.args, Expr(:for, ex.args[1]))
                loopbody, tmp_newindx, tmp_arraydecl = _newfnbody( ex.args[2], fnargs, d_indx )
                push!(newfnbody.args[end].args, loopbody)
                append!(v_newindx, tmp_newindx)
                append!(v_arraydecl, tmp_arraydecl)
            elseif ex_head == :macrocall
                # Deal with `Threads.@threads` and `@threads` cases
                if ex.args[1] in [Expr(:., :Threads, QuoteNode(Symbol("@threads"))), Symbol("@threads")]
                    push!(newfnbody.args, Expr(:macrocall, ex.args[1]))
                    # Although here `ex.args[2]` is a `LineNumberNode`,
                    # we add it to `newfnbody` because `@threads` call expressions require 3 args
                    push!(newfnbody.args[end].args, ex.args[2])
                    # Since `@threads` is called before a `for` loop, we deal
                    # with `ex.args[3]` as a `for` loop
                    push!(newfnbody.args[end].args, Expr(:for, ex.args[3].args[1]))
                    atthreadsbody, tmp_newindx, tmp_arraydecl = _newfnbody( ex.args[3].args[2], fnargs, d_indx )
                    push!(newfnbody.args[end].args[end].args, atthreadsbody)
                    append!(v_newindx, tmp_newindx)
                    append!(v_arraydecl, tmp_arraydecl)
                else
                    # If macro not implemented, throw an `ArgumentError`
                    throw(ArgumentError("Macro $(ex.args[1]) is not yet implemented"))
                end
            elseif ex_head == :if
                # The first argument of an `if` expression is the condition, the
                # second one is the block the condition is true, and the third
                # is the `else`-block or an `ifelse`-block.
                push!(newfnbody.args, Expr(:if, ex.args[1]))
                for exx in ex.args[2:end]
                    if exx.head == :elseif
                        exxx = Expr(:block, Expr(:if, exx.args[1].args[2], exx.args[2:end]...))
                    else #if exx.head == :block  # `then` or `else` blocks
                        exxx = exx
                    end
                    ifbody, tmp_newindx, tmp_arraydecl = _newfnbody(exxx, fnargs, d_indx)
                    push!(newfnbody.args[end].args, ifbody)
                    append!(v_newindx, tmp_newindx)
                    append!(v_arraydecl, tmp_arraydecl)
                end
            elseif ex_head == :(=) || ex_head == :call  # assignements or function calls

                ex_lhs = ex.args[1]
                ex_rhs = ex.args[2]

                # Case of explicit declaration of Array or Vector
                # TO-DO: Include cases where the definition uses `similar`
                if !isempty(findex(:(_Array{_TT...}), ex)) ||
                        !isempty(findex(:(Vector{_TT...}), ex))
                    push!(v_arraydecl, ex_lhs)
                    push!(newfnbody.args, ex)
                    continue
                end

                # Case of multiple assignments, e.g., ex = :((σ, ρ, β) = par)
                # Distinguishes if the RHS is the parameter or the dependent vars
                if isa(ex_lhs, Expr)
                    if ex_lhs.head == :tuple && length(fnargs) == 4 && isa(ex_rhs, Symbol)
                        # Construct new expressions
                        if ex_rhs == fnargs[3]  # params
                            for i in eachindex(ex_lhs.args)
                                ex_indx = Expr(:local, Expr(:(=), ex_lhs.args[i], Expr(:ref, ex_rhs, i)))
                                push!(newfnbody.args, ex_indx)
                            end
                        elseif ex_rhs == fnargs[2] # `x`
                            for i in eachindex(ex_lhs.args)
                                ex_indx = Expr(:(=), ex_lhs.args[i], Expr(:ref, ex_rhs, i))
                                push!(newfnbody.args, ex_indx)
                            end
                        else
                            throw(ArgumentError("Assignment no implemented in $(ex)"))
                        end
                        #
                    end
                    continue
                end

                # Unfold AST graph
                nex = deepcopy(ex)
                # try
                #     nex = to_expr(ExGraph(simplify(ex)))
                # catch
                #     # copy `ex` as it is, if it is not "recognized"
                #     push!(newfnbody.args, ex)
                #     continue
                # end
                nex = to_expr(ExGraph(simplify(ex)))
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
                    (isindx_lhs || vars_nex[1] != ex_lhs) && push!(v_newindx, vars_nex[1])
                end
            elseif (ex_head == :local) || (ex_head == :continue) || (ex_head == :break)
                # If declared as `local` or `continue`, copy `ex` as it is.
                # In some cases this, using `local` helps performance. Very
                # useful for including (numeric) constants
                push!(newfnbody.args, ex)
                #
            else
                # If not implemented, throw an `ArgumentError`
                throw(ArgumentError("$(ex_head) is not yet implemented; $(typeof(ex))"))
            end
            #
        elseif isa(ex, LineNumberNode)
            continue  # Ignore `LineNumberNode`s
        else
            # In any other case, throw an `ArgumentError`
            throw(ArgumentError("$ex is not an `Expr`; $(typeof(ex))"))
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

    # Magic happens HERE
    # Each line of ex (ex.args) is parsed separately
    for (i, aa) in enumerate(ex.args)

        # Treat for loops, @threads for loops, blocks and if blocks separately
        if (aa.head == :for)
            push!(preex.args, Expr(:for, aa.args[1]))
            _parse_newfnbody!(aa, preex.args[end],
                v_vars, v_assign, d_indx, v_newindx, v_arraydecl, true)
            #
        elseif (aa.head == :macrocall && aa.args[1] in [Expr(:., :Threads, QuoteNode(Symbol("@threads"))), Symbol("@threads")])
            push!(preex.args, Expr(:macrocall, aa.args[1]))
            push!(preex.args[end].args, aa.args[2])
            push!(preex.args[end].args, Expr(:for, aa.args[3].args[1]))
            _parse_newfnbody!(aa.args[3], preex.args[end].args[end],
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
                # Replace expression
                _replace_expr!(ex, preex, i, aa_lhs, aa_rhs, v_vars, d_indx, v_newindx)

                # Remove new aa_lhs declaration from `ex`; it is already declared
                in(aa_lhs, v_arraydecl) && push!(indx_rm, i)
                #
            elseif isa(aa_rhs, Symbol) # occurs when there is a simple assignment

                bb = subs(aa, Dict(aa_rhs => :(identity($aa_rhs))))
                bb_lhs = bb.args[1]
                bb_rhs = bb.args[2]
                # Replace expression
                _replace_expr!(ex, preex, i, bb_lhs, bb_rhs, v_vars, d_indx, v_newindx)
                #
            elseif isa(aa_rhs, Number)  # case of numeric values
                push!(v_assign, aa_lhs => aa_rhs)
                push!(indx_rm, i)
                #
            else # needed?
                error("Either $aa or $typeof(aa_rhs[2]) are different from `Expr`, `Symbol` or `Number`")
                #
            end
            #
        elseif aa.head == :local
            # If declared as `local`, copy `ex` as it is, and delete it from
            # the recursion body
            push!(preex.args, aa)
            push!(indx_rm, i)   # delete associated expr in body function
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
        fnexpr = subs(fnexpr, Dict(:_res => newvar, :_arg1 => newarg1, :_k => :ord))
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
`v_preamb` is for bookkeeping the introduced variables.

"""
function _defs_preamble!(preamble::Expr, fnargs,
        d_indx, v_newindx, v_arraydecl, v_preamb, d_decl, inloop::Bool=false,
        ex_aux::Expr = Expr(:block,))

    # Initializations
    defspreamble = Expr[]

    for (i, ex) in enumerate(preamble.args)

        if isa(ex, Expr)

            # Treat block, for loops, @threads for loops, if separately
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
            elseif (ex.head == :macrocall  && ex.args[1] in [Expr(:., :Threads, QuoteNode(Symbol("@threads"))), Symbol("@threads")])
                push!(ex_aux.args, ex.args[3].args[1])
                newdefspr = _defs_preamble!(ex.args[3].args[2], fnargs, d_indx,
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

                    # NOTE: Uses the size of the var with more indices
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
            throw(ArgumentError("$ex is not an `Expr`; $(typeof(ex))"))
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
    if ll == 3

        rec_preamb = sanitize( :( $(fnargs[1])[1] = $(retvar)[0] ) )
        rec_fnbody = sanitize( :( $(fnargs[1])[ordnext] = $(retvar)[ord]/ordnext ) )
        #
    elseif ll == 4

        retvar = fnargs[1]
        rec_preamb = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[2] = $(retvar)[__idx].coeffs[1]
            end
        ))
        rec_fnbody = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[ordnext+1] =
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
`@taylorize expr`

This macro `eval`s the function given by `expr` and defines a new
method of [`jetcoeffs!`](@ref) which is specialized on that
function. Integrating via [`taylorinteg`](@ref) of
[`lyap_taylorinteg`](@ref) after using the macro yields better performance.

See the [documentation](@ref taylorize) for more details and limitations.

!!! warning
    This macro is on an experimental stage; check the integration
    results carefully.

"""
macro taylorize( ex )
    nex = _make_parsed_jetcoeffs(ex)
    esc(quote
        $ex  # evals to calling scope the passed function
        $nex # evals the new method of `TaylorIntegration.jetcoeffs!`
    end)
end
