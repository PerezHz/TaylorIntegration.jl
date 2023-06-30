# This file is part of the TaylorIntegration.jl package; MIT licensed

# Load necessary components of Espresso
using Espresso: subs, simplify, ExGraph, ExH, to_expr, sanitize, genname,
    find_vars, findex, find_indices, isindexed


"""
    BookKeeping

Mutable struct that contains all the bookkeeping vectors/dictionaries used within
`_make_parsed_jetcoeffs`:
    - `d_indx`     : Dictionary mapping new variables (symbols) to old (perhaps indexed) symbols
    - `d_assign`   : Dictionary with the numeric assignments (that are substituted)
    - `d_decl`     : Dictionary declared arrays
    - `v_newvars`  : Symbols of auxiliary indexed vars
    - `v_arraydecl`: Symbols which are explicitly declared as Array or Vector
    - `v_array1`   : Symbols which are explicitly declared as Array{Taylor1{T},1}
    - `v_array2`   : Symbols which are explicitly declared as Array{Taylor1{T},2}
    - `v_array3`   : Symbols which are explicitly declared as Array{Taylor1{T},3}
    - `v_array4`   : Symbols which are explicitly declared as Array{Taylor1{T},4}
    - `v_preamb`   : Symbols or Expr used in the preamble (declarations, etc)
    - `retvar`     : *Guessed* returned variable, which defines the LHS of the ODEs

"""
mutable struct BookKeeping
    d_indx   :: Dict{Symbol, Expr}
    d_assign :: Dict{Union{Symbol,Expr}, Number}
    d_decl   :: Dict{Symbol, Expr}
    v_newvars   :: Vector{Symbol}
    v_arraydecl :: Vector{Symbol}
    v_array1 :: Vector{Symbol}
    v_array2 :: Vector{Symbol}
    v_array3 :: Vector{Symbol}
    v_array4 :: Vector{Symbol}
    v_preamb :: Vector{Union{Symbol,Expr}}
    retvar   :: Symbol

    function BookKeeping()
        return new(Dict{Symbol, Expr}(), Dict{Union{Symbol,Expr}, Number}(),
            Dict{Symbol, Expr}(), Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
            Union{Symbol,Expr}[], :nothing)
    end
end


"""
    RetAlloc{Taylor1{T}}

Struct related to the returned variables that are pre-allocated when
`@taylorize` is used.
    - `v0`   : Vector{Taylor1{T}}
    - `v1`   : Vector{Vector{Taylor1{T}}}

"""
struct RetAlloc{T <: Number}
    v0 :: Array{T,1}
    v1 :: Vector{Array{T,1}}
    v2 :: Vector{Array{T,2}}
    v3 :: Vector{Array{T,3}}
    v4 :: Vector{Array{T,4}}

    function RetAlloc{T}() where {T}
        v1 = Array{T,1}(undef, 0)
        return new(v1, [v1], [Array{T,2}(undef, 0, 0)], [Array{T,3}(undef, 0, 0, 0)],
            [Array{T,4}(undef, 0, 0, 0, 0)])
    end

    function RetAlloc{T}(v0::Array{T,1}, v1::Vector{Array{T,1}},
            v2::Vector{Array{T,2}}, v3::Vector{Array{T,3}}, v4::Vector{Array{T,4}}) where {T}
        return new(v0, v1, v2, v3, v4)
    end
end


"""
`inbookkeeping(v, bkkeep::BookKeeping)`

Checks if `v` is declared in `bkkeep`, considering the `d_indx`, `v_newvars` and
`v_arraydecl` fields.

"""
@inline inbookkeeping(v, bkkeep::BookKeeping) = (v ∈ keys(bkkeep.d_indx) ||
    v ∈ bkkeep.v_newvars || v ∈ bkkeep.v_arraydecl)


# Constants to create the structure of the new jetcoeffs! and _allocate_jetcoeffs! methods.
# The (irrelevant) `nothing` below is there to have a `:block` Expr; it is deleted later
const _HEAD_PARSEDFN_SCALAR = sanitize(:(
    function TaylorIntegration.jetcoeffs!(::Val{__fn}, __tT::Taylor1{_T}, __x::Taylor1{_S},
            __params, __ralloc::TaylorIntegration.RetAlloc{Taylor1{_S}}) where {_T<:Real, _S<:Number}
        order = __tT.order
        nothing
    end)
);

const _HEAD_PARSEDFN_VECTOR = sanitize(:(
    function TaylorIntegration.jetcoeffs!(::Val{__fn}, __tT::Taylor1{_T},
            __x::AbstractArray{Taylor1{_S}, _N}, __dx::AbstractArray{Taylor1{_S}, _N},
            __params, __ralloc::TaylorIntegration.RetAlloc{Taylor1{_S}}) where {_T<:Real, _S<:Number, _N}
        order = __tT.order
        nothing
    end)
);

const _HEAD_ALLOC_TAYLOR1_SCALAR = sanitize(:(
    function TaylorIntegration._allocate_jetcoeffs!( ::Val{__fn}, __tT::Taylor1{_T},
            __x::Taylor1{_S}, __params) where {_T<:Real, _S<:Number}
        order = __tT.order
        nothing
    end)
);

const _HEAD_ALLOC_TAYLOR1_VECTOR = sanitize(:(
    function TaylorIntegration._allocate_jetcoeffs!( ::Val{__fn}, __tT::Taylor1{_T},
            __x::AbstractArray{Taylor1{_S}, _N}, __dx::AbstractArray{Taylor1{_S}, _N},
            __params) where {_T<:Real, _S<:Number, _N}
        order = __tT.order
        nothing
    end)
);

# Constants for the initial declaration and initialization of arrays
const _DECL_ARRAY = sanitize( Expr(:block,
    :(__var1 = Array{Taylor1{_S}}(undef, __var2)),
    :(__var1 .= Taylor1( zero(_S), order )))
);


"""
`_make_parsed_jetcoeffs( ex )`

This function constructs the expressions of two new methods, the first equivalent to the
differential equations (jetcoeffs!), which exploits the mutating functions of TaylorSeries.jl,
and the second one (_allocate_jetcoeffs) preallocates any auxiliary `Taylor1` or
`Vector{Taylor1{T}}` needed.

"""
function _make_parsed_jetcoeffs(ex::Expr)
    # Extract the name, args and body of the function
    # `fn` name of the function having the ODEs
    # `fnargs` arguments of the function
    # `fnbody` future transformed function body
    fn, fnargs, fnbody = _extract_parts(ex)

    # Set up the Expr for the new functions
    new_jetcoeffs, new_allocjetcoeffs = _newhead(fn, fnargs)

    # Transform the graph representation of the body of the functions:
    # defspreamble: inicializations used for the zeroth order (preamble)
    # defsprealloc: definitions (declarations) of auxiliary Taylor1's
    # fnbody: transformed function body, using mutating functions from TaylorSeries;
    #         used later within the recursion loop
    # bkkeep: book-keeping structure having info of the variables
    defspreamble, defsprealloc, fnbody, bkkeep = _preamble_body(fnbody, fnargs)

    # Create body of recursion loop; temporary assignements may be needed.
    # rec_preamb: recursion loop for the preamble (first order correction)
    # rec_fnbody: recursion loop for the body-function (recursion loop for higher orders)
    rec_preamb, rec_fnbody = _recursionloop(fnargs, bkkeep)
    rec_preamb = Expr(:macrocall, Symbol("@inbounds"),
        LineNumberNode(@__LINE__, Symbol(@__FILE__)), rec_preamb)
    rec_fnbody = Expr(:macrocall, Symbol("@inbounds"),
        LineNumberNode(@__LINE__, Symbol(@__FILE__)), rec_fnbody)

    # Expr for the for-loop block for the recursion (of the `x` variable)
    forloopblock = Expr(:for, :(ord = 1:order-1), Expr(:block, :(ordnext = ord + 1)) )
    forloopblock = Expr(:macrocall, Symbol("@inbounds"),
        LineNumberNode(@__LINE__, Symbol(@__FILE__)), forloopblock)
    # Add rec_fnbody to forloopblock
    # push!(forloopblock.args[2].args, fnbody.args[1].args..., rec_fnbody)
    push!(forloopblock.args[3].args[2].args, fnbody.args[1].args..., rec_fnbody)

    # Add preamble and recursion body to `new_jetcoeffs`
    push!(new_jetcoeffs.args[2].args, defspreamble..., rec_preamb)

    # Push preamble and forloopblock to `new_jetcoeffs` and return line
    push!(new_jetcoeffs.args[2].args, forloopblock, Meta.parse("return nothing"))

    # Split v_arraydecl according to the number of indices
    _split_arraydecl!(bkkeep)

    # Add allocated variable definitions to `new_jetcoeffs`, to make it more human readable
    _allocated_defs!(new_jetcoeffs, bkkeep)

    # Define the expressions of the returned vectors in `new_allocjetcoeffs`
    push!(new_allocjetcoeffs.args[2].args, defsprealloc...)

    # Define returned expression for `new_allocjetcoeffs`
    ret_ret = _returned_expr(bkkeep)

    # Add return line to `new_allocjetcoeffs`
    push!(new_allocjetcoeffs.args[2].args, ret_ret)

    # Rename variables in the calling form of the new methods
    if length(fnargs) == 3
        new_jetcoeffs = subs(new_jetcoeffs, Dict(:__tT => fnargs[3],
            :__params => fnargs[2], :(__x) => fnargs[1], :(__fn) => fn ))
        new_allocjetcoeffs = subs(new_allocjetcoeffs,
            Dict(:__tT => fnargs[3], :__params => fnargs[2],
                :(__x) => fnargs[1], :(__fn) => fn ))

    elseif length(fnargs) == 4
        new_jetcoeffs = subs(new_jetcoeffs, Dict(:__tT => fnargs[4],
            :__params => fnargs[3], :(__x) => fnargs[2], :(__dx) => fnargs[1],
            :(__fn) => fn ))
        new_allocjetcoeffs = subs(new_allocjetcoeffs,
            Dict(:__tT => fnargs[4], :__params => fnargs[3],
                :(__x) => fnargs[2], :(__dx) => fnargs[1], :(__fn) => fn ))

    # else
    #     # A priori this is not needed
    #     throw(ArgumentError("Wrong number of arguments in `fnargs`"))
    end

    return new_jetcoeffs, new_allocjetcoeffs
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
`_capture_fn_args_body!(ex, dd::Dict{Symbol, Any})`

Captures the name of a function, arguments, body and other properties,
returning them as the values of the dictionary `dd`, which is updated
in place.

"""
function _capture_fn_args_body!(ex::Expr, dd::Dict{Symbol, Any} = Dict())
    exh = ExH(ex)
    if exh.head == :(=) || exh.head == :function
        _capture_fn_args_body!.(ex.args, Ref(dd))
    elseif exh.head == :call
        push!(dd, Pair(:fname, exh.args[1]))
        push!(dd, Pair(:fnargs, exh.args[2:end]))
    elseif exh.head == :block
        push!(dd, Pair(:fnbody, ex))
    elseif exh.head == :where
        push!(dd, Pair(:where, ex.args[2:end]))
        _capture_fn_args_body!(ex.args[1], dd)
    end

    return nothing
end



"""
`_newhead(fn, fnargs)`

Creates the head of the new method of `jetcoeffs!` and `_allocate_jetcoeffs`.
`fn` is the name of the passed function and `fnargs` is a vector with its arguments
defning the function (which are either three or four).

"""
function _newhead(fn, fnargs)
    # Construct common elements of the new expression
    if length(fnargs) == 3
        newfunction = copy(_HEAD_PARSEDFN_SCALAR)
        newdeclfunc = copy(_HEAD_ALLOC_TAYLOR1_SCALAR)

    elseif length(fnargs) == 4
        newfunction = copy(_HEAD_PARSEDFN_VECTOR)
        newdeclfunc = copy(_HEAD_ALLOC_TAYLOR1_VECTOR)

    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    # Delete the irrelevant `nothing`
    pop!(newfunction.args[2].args)
    pop!(newdeclfunc.args[2].args)

    return newfunction, newdeclfunc
end



"""
`_preamble_body(fnbody, fnargs)`

Returns expressions for the preamble, the declaration of
arrays, the body and the bookkeeping struct, which will be used to build
the new functions. `fnbody` is the expression with the body of
the original function (already adapted), `fnargs` is a vector of symbols
of the original diferential equations function.

"""
function _preamble_body(fnbody, fnargs)
    # Inicialize BookKeeping struct
    bkkeep = BookKeeping()

    # Rename vars to have the body in non-indexed form; bkkeep has different entries
    # for bookkeeping variables/symbolds, including indexed ones
    fnbody, bkkeep.d_indx = _rename_indexedvars(fnbody)

    # Create `newfnbody` which corresponds to `fnbody`, cleaned (without irrelevant comments)
    # and with all new variables in place; bkkeep.d_indx is updated
    newfnbody = _newfnbody!(fnbody, fnargs, bkkeep)

    # Parse `newfnbody` and create `prepreamble` and `prealloc`, updating `bkkeep`.
    # These objects use the mutating functions from TaylorSeries.
    preamble = Expr(:block,)
    prealloc = Expr(:block,)
    _parse_newfnbody!(newfnbody, preamble, prealloc, bkkeep, false)
    # Get rid of the initial `:block`
    preamble = preamble.args[1]
    prealloc = prealloc.args[1]

    # Substitute numeric assignments in new function's body
    newfnbody = subs(newfnbody, bkkeep.d_assign)
    preamble = subs(preamble, bkkeep.d_assign)
    prealloc = subs(prealloc, bkkeep.d_assign)

    # Include the assignement of indexed auxiliary variables
    defsprealloc = _defs_allocs!(prealloc, fnargs, bkkeep, false)
    preamble = subs(preamble, bkkeep.d_indx)
    defspreamble = Expr[preamble.args...]
    # Bring back substitutions
    newfnbody = subs(newfnbody, bkkeep.d_indx)

    # Define retvar; for scalar eqs is the last entry included in v_newvars
    bkkeep.retvar = length(fnargs) == 3 ? subs(bkkeep.v_newvars[end], bkkeep.d_indx) : fnargs[1]

    return defspreamble, defsprealloc, newfnbody, bkkeep
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
    return new_fnbody, Dict{Symbol, Expr}(v => k for (k, v) in st)
end



"""
`_newfnbody!(fnbody, fnargs, bkkeep)`

Returns a new (modified) body of the function, a priori unfolding
the expression graph (AST) as unary and binary calls, and updates
the bookkeeping structure bkkeep.

"""
function _newfnbody!(fnbody, fnargs, bkkeep::BookKeeping)
    # `fnbody` is assumed to be a `:block`-Expr
    newfnbody = Expr(:block,)

    # Local definition for possible args for `@threads` call
    local v_threads_args = (Expr(:., :Threads, QuoteNode(Symbol("@threads"))),
        Symbol("@threads"))

    # Magic happens HERE!!
    # Each line of fnbody (fnbody.args) is parsed separately
    for (i, ex) in enumerate(fnbody.args)
        if isa(ex, Expr)
            ex_head = ex.head

            # The `return` statement is treated separately
            if ex.head == :return
                @assert length(ex.args) == 1  # one sole returned value

                # Do nothing if it is `nothing` or if its in fnargs
                (ex.args[1] == :nothing || ex.args[1] in fnargs) && continue

                # Otherwise, create a new assignment, and process it
                new_aux_var = genname()
                ex = :($new_aux_var = $(ex.args[1]))
                ex_head = ex.head
            end

            # Treat `for` loops, `Threads.@threads for` and `if` blocks separately
            if ex_head == :block
                newblock = _newfnbody!(ex, fnargs, bkkeep)
                push!(newfnbody.args, newblock )

            elseif ex_head == :for
                push!(newfnbody.args, Expr(:for, ex.args[1]))
                loopbody = _newfnbody!( ex.args[2], fnargs, bkkeep )
                push!(newfnbody.args[end].args, loopbody)

            elseif ex_head == :macrocall
                # TODO: Deal with `@inbounds`
                # Deal with `Threads.@threads` and `@threads` cases
                if ex.args[1] in v_threads_args
                    push!(newfnbody.args, Expr(:macrocall, ex.args[1]))
                    # Although here `ex.args[2]` is a `LineNumberNode`,
                    # we add it to `newfnbody` because `@threads` call expressions
                    # require 3 args
                    push!(newfnbody.args[end].args, ex.args[2])
                    # Since `@threads` is called before a `for` loop, we deal
                    # with `ex.args[3]` as a `for` loop
                    push!(newfnbody.args[end].args, Expr(:for, ex.args[3].args[1]))

                    atthreadsbody = _newfnbody!( ex.args[3].args[2], fnargs, bkkeep )
                    push!(newfnbody.args[end].args[end].args, atthreadsbody)

                else
                    # If macro is not implemented, throw an `ArgumentError`
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
                    else
                        exxx = exx
                    end

                    ifbody = _newfnbody!(exxx, fnargs, bkkeep)
                    push!(newfnbody.args[end].args, ifbody)
                end

            elseif ex_head == :(=) || ex_head == :call
                # Assignements or function calls
                ex_lhs = ex.args[1]
                ex_rhs = ex.args[2]

                # TODO: Include cases where the definition uses `similar`
                # Case of explicit declaration of Array or Vector
                if !isempty(findex(:(_Array{_TT...}), ex)) ||
                        !isempty(findex(:(Vector{_TT...}), ex))
                    inbookkeeping(ex_lhs, bkkeep) || push!(bkkeep.v_arraydecl, ex_lhs)
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
                                ex_indx = Expr(:local, Expr(:(=), ex_lhs.args[i],
                                    Expr(:ref, ex_rhs, i)))
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
                    end
                    continue
                end

                # Unfold AST graph
                # nex = deepcopy(ex)
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
                isindx_lhs = haskey(bkkeep.d_indx, ex_lhs)
                for nexargs in nex.args[2:end]
                    haskey(bkkeep.d_indx, nexargs.args[1]) && continue
                    vars_nex = find_vars(nexargs)

                    # any(haskey.(bkkeep.d_indx, Ref(vars_nex))) &&
                    #     !in(vars_nex[1], bkkeep.v_newvars) &&
                    (isindx_lhs || vars_nex[1] != ex_lhs) &&
                        !inbookkeeping(vars_nex[1], bkkeep) &&
                            push!(bkkeep.v_newvars, vars_nex[1])
                end

            elseif (ex_head == :local) || (ex_head == :continue) || (ex_head == :break)
                # If declared as `local`, or it is a `continue` or `break`, copy `ex` as it is.
                # In some cases using `local` helps performance. Very
                # useful for including (numeric) constants
                push!(newfnbody.args, ex)

            else
                # If not implemented, throw an `ArgumentError`
                throw(ArgumentError("$(ex_head) is not yet implemented; $(typeof(ex))"))
            end

        elseif isa(ex, LineNumberNode)
            continue  # Ignore `LineNumberNode`s

        else
            # In any other case, throw an `ArgumentError`
            throw(ArgumentError("$ex is not an `Expr`; $(typeof(ex))"))
        end
    end

    # Needed, if `newfnbody` consists of a single assignment (unary call)
    if newfnbody.head == :(=)
        newfnbody = Expr(:block, newfnbody)
    end

    return newfnbody
end



"""
_parse_newfnbody!(ex::Expr, preex::Expr, prealloc::Expr, bkkeep::BookKeeping, inloop::Bool)

Parses `ex` (the new body of the function) replacing the expressions
to use the mutating functions of TaylorSeries, and building the preamble
`preex` and `prealloc` expressions. This is done by traversing recursively (again)
the args of `ex`, updating the bookkeeping struct `bkkeep`, in particular
the fieldnames `v_newvars` and `d_assign`.

"""
function _parse_newfnbody!(ex::Expr, preex::Expr, prealloc::Expr, bkkeep::BookKeeping,
        inloop::Bool)
    # Numeric assignements to be deleted
    indx_rm = Int[]

    # Definition for possible args for `@threads` call
    local v_threads_args = (Expr(:., :Threads, QuoteNode(Symbol("@threads"))), Symbol("@threads"))

    # Magic happens HERE
    # Each line of `ex` (i.e., `ex.args`) is parsed separately
    for (i, aa) in enumerate(ex.args)
        # Treat for-loops, @threads macro, blocks and if-blocks separately
        # (through `inloop=true`)
        if (aa.head == :for)
            push!(preex.args, Expr(:for, aa.args[1]))
            push!(prealloc.args, Expr(:for, aa.args[1]))

            _parse_newfnbody!(aa, preex.args[end], prealloc.args[end], bkkeep, true)

        elseif (aa.head == :macrocall && aa.args[1] in v_threads_args)
            push!(preex.args, Expr(:macrocall, aa.args[1]))
            push!(preex.args[end].args, aa.args[2])
            push!(preex.args[end].args, Expr(:for, aa.args[3].args[1]))
            push!(prealloc.args, Expr(:macrocall, aa.args[1]))
            push!(prealloc.args[end].args, aa.args[2])
            push!(prealloc.args[end].args, Expr(:for, aa.args[3].args[1]))

            _parse_newfnbody!(aa.args[3], preex.args[end].args[end], prealloc.args[end].args[end],
                bkkeep, true)

        elseif (aa.head == :block)
            push!(preex.args, Expr(:block))
            push!(prealloc.args, Expr(:block))

            # `inloop` needs to be `false` here
            _parse_newfnbody!(aa, preex.args[end], prealloc.args[end], bkkeep, false)

        elseif (aa.head == :if)
            push!(preex.args, Expr(:if, aa.args[1]))
            push!(prealloc.args, Expr(:if, aa.args[1]))

            for exx in aa.args[2:end]
                push!(preex.args[end].args, Expr(:block))
                push!(prealloc.args[end].args, Expr(:block))
                _parse_newfnbody!(exx, preex.args[end].args[end],
                    prealloc.args[end].args[end], bkkeep, inloop)
            end

        elseif (aa.head == :(=))
            # Skip parsing the first for-loop expression (`inloop=true`), which is of
            # the form `:(i = ...)`
            i == 1 && inloop && continue

            # Main case
            aa_lhs = aa.args[1]
            aa_rhs = aa.args[2]

            # Replace expressions when needed, and bookkeeping
            if isa(aa_rhs, Expr)
                (aa_rhs.args[1] == :eachindex || aa_rhs.head == :(:)) && continue

                # Replace expression
                _replace_expr!(ex, preex, prealloc, i, aa_lhs, aa_rhs, bkkeep)

                # Remove new aa_lhs declaration from `ex`; it is already declared
                in(aa_lhs, bkkeep.v_arraydecl) && push!(indx_rm, i)

            elseif isa(aa_rhs, Symbol) # occurs when there is a simple assignment
                bb = subs(aa, Dict(aa_rhs => :(identity($aa_rhs))))
                bb_lhs = bb.args[1]
                bb_rhs = bb.args[2]

                # Replace expression
                _replace_expr!(ex, preex, prealloc, i, bb_lhs, bb_rhs, bkkeep)

            elseif isa(aa_rhs, Number)  # case of numeric values
                push!(bkkeep.d_assign, aa_lhs => aa_rhs)
                push!(indx_rm, i)
                if aa_lhs ∈ bkkeep.v_newvars
                    iis = findall(x->x==aa_lhs, bkkeep.v_newvars)
                    deleteat!(bkkeep.v_newvars, iis)
                end

            else # needed?
                error("Either $aa or $typeof(aa_rhs[2]) are not an `Expr`, `Symbol` or `Number`")
            end

        elseif aa.head == :local
            # If declared as `local`, copy `ex` as it is, and delete it from the recursion body
            push!(preex.args, aa)
            push!(prealloc.args, aa)
            push!(indx_rm, i)   # delete associated expr in body function

        else
            # Unrecognized head; pass the expression as it is
            push!(preex.args, aa)
            push!(prealloc.args, aa)
        end
    end

    # Delete trivial numeric assignement statements
    isempty(indx_rm) || deleteat!(ex.args, indx_rm)

    return nothing
end



"""
`_replace_expr!(ex::Expr, preex::Expr, , prealloc::Expr, i::Int, aalhs, aarhs,
    bkkeep::BookKeeping)`

Replaces the calls in `ex.args[i]`, and updates `preex` and `prealloc` with the
appropriate expressions, based on the the LHS (`aalhs`) and RHS (`aarhs`) of the
base assignment. The bookkeeping struct is updated (`v_newvars`) within `_replacecalls!`.
`d_indx` is used to bring back the indexed variables.

"""
function _replace_expr!(ex::Expr, preex::Expr, prealloc::Expr, i::Int,
        aalhs, aarhs, bkkeep::BookKeeping)
    # Replace calls
    fnexpr, def_fnexpr, auxfnexpr, def_alloc, aux_alloc = _replacecalls!(bkkeep, aarhs, aalhs)

    # Bring back indexed variables
    fnexpr     = subs(fnexpr, bkkeep.d_indx)
    def_fnexpr = subs(def_fnexpr, bkkeep.d_indx)
    auxfnexpr  = subs(auxfnexpr, bkkeep.d_indx)
    def_alloc  = subs(def_alloc, bkkeep.d_indx)
    aux_alloc  = subs(aux_alloc, bkkeep.d_indx)

    # Update `ex`, `preex` and `prealloc`
    if (fnexpr != def_fnexpr)
        ex.args[i] = fnexpr
    end

    if def_fnexpr.head != :nothing
        if def_fnexpr.head == :block
            append!(preex.args, def_fnexpr.args)
        else
            push!(preex.args, def_fnexpr)
        end
    end

    if def_alloc.head != :nothing
        if def_alloc.head == :block
            append!(prealloc.args, def_alloc.args)
        else
            push!(prealloc.args, def_alloc)
        end
    end

    # Same for the auxiliary expressions
    if auxfnexpr.head != :nothing
        if auxfnexpr.head == :block
            append!(preex.args, auxfnexpr.args)
        else
            push!(preex.args, auxfnexpr)
        end
        aux_alloc.head == :nothing || push!(prealloc.args, aux_alloc)
    end

    return nothing
end



"""
`_replacecalls!(bkkeep, fnold, newvar)`

Replaces the symbols of unary and binary calls
of the expression `fnold`, which defines `newvar`,
by the mutating functions in TaylorSeries.jl.
The vector `bkkeep.v_vars` is updated if new auxiliary
variables are introduced (bookkeeping).

"""
function _replacecalls!(bkkeep::BookKeeping, fnold::Expr, newvar::Symbol)
    ll = length(fnold.args)
    dcall = fnold.args[1]
    newarg1 = fnold.args[2]

    # If call is not in the mutating functions dictionaries, copy original one
    # to def_fnexpr and def_alloc, except if it is an Array/Vector definition
    if !( in(dcall, keys(TaylorSeries._dict_unary_calls)) ||
          in(dcall, keys(TaylorSeries._dict_binary_calls)) )

        fnexpr = :($newvar = $fnold)
        if isa(dcall, Expr) && (dcall.args[1] == :Array || dcall.args[1] == :Vector)
            def_fnexpr = Expr(:nothing)
            inbookkeeping(newvar, bkkeep) || push!(bkkeep.v_arraydecl, newvar)
        else
            def_fnexpr = fnexpr
            inbookkeeping(newvar, bkkeep) || push!(bkkeep.v_newvars, newvar)
        end
        def_alloc = fnexpr

        return fnexpr, def_fnexpr, Expr(:nothing), def_alloc, Expr(:nothing)
    end

    # Bookkeeping
    inbookkeeping(newvar, bkkeep) || push!(bkkeep.v_newvars, newvar)

    # Initializing
    def_alloc = Expr(:nothing)
    aux_alloc = Expr(:nothing)

    if ll == 2
        # Unary call
        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_unary_calls[dcall]
        fnexpr = subs(fnexpr, Dict(:_res => newvar, :_arg1 => newarg1, :_k => :ord))

        def_alloc = :( _res = Taylor1($(def_fnexpr.args[2]), order) )
        def_alloc = subs(def_alloc,
            Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))), :_k => :ord))

        def_fnexpr = Expr(:block,
            :(_res.coeffs[1] = $(def_fnexpr.args[2])),
            :(_res.coeffs[2:order+1] .= zero(_res.coeffs[1])) )
        def_fnexpr = subs(def_fnexpr,
            Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))), :_k => :ord))
        # def_fnexpr = Expr(:block,
        #     :(_res[0] = $(def_fnexpr.args[2])),
        #     :(_res[1:order] .= zero(_res[0])) )
        # def_fnexpr = subs(def_fnexpr,
        #     Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))), :_k => :ord))

        # Auxiliary expression
        if aux_fnexpr.head != :nothing
            newaux = genname()

            aux_alloc = :( _res = Taylor1($(aux_fnexpr.args[2]), order) )
            aux_alloc = subs(aux_alloc,
                Dict(:_res => newaux, :_arg1 => :(constant_term($(newarg1))), :_aux => newaux))

            aux_fnexpr = Expr(:block,
                :(_res.coeffs[1] = $(aux_fnexpr.args[2])),
                :(_res.coeffs[2:order+1] .= zero(_res.coeffs[1])) )
            aux_fnexpr = subs(aux_fnexpr,
                Dict(:_res => newaux, :_arg1 => :(constant_term($(newarg1))), :_aux => newaux))

            fnexpr = subs(fnexpr, Dict(:_aux => newaux))
            if newvar ∈ bkkeep.v_arraydecl
                push!(bkkeep.v_arraydecl, newaux)
            else
                push!(bkkeep.v_newvars, newaux)
            end
        end

    elseif ll == 3
        # Binary call; no auxiliary expressions needed
        newarg2 = fnold.args[3]

        # Replacements
        fnexpr, def_fnexpr, aux_fnexpr = TaylorSeries._dict_binary_calls[dcall]
        fnexpr = subs(fnexpr,
            Dict(:_res => newvar, :_arg1 => newarg1, :_arg2 => newarg2, :_k => :ord))

        def_alloc = :(_res = Taylor1($(def_fnexpr.args[2]), order) )
        def_alloc = subs(def_alloc, Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))),
            :_arg2 => :(constant_term($(newarg2))), :_k => :ord) )

        def_fnexpr = Expr(:block,
            :(_res.coeffs[1] = $(def_fnexpr.args[2])),
            :(_res.coeffs[2:order+1] .= zero(_res.coeffs[1])) )
        def_fnexpr = subs(def_fnexpr,
            Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))),
                :_arg2 => :(constant_term($(newarg2))), :_k => :ord))
        # def_fnexpr = Expr(:block,
        #     :(_res[0] = $(def_fnexpr.args[2])),
        #     :(_res[1:order] .= zero(_res[0])) )
        # def_fnexpr = subs(def_fnexpr,
        #     Dict(:_res => newvar, :_arg1 => :(constant_term($(newarg1))),
        #         :_arg2 => :(constant_term($(newarg2))), :_k => :ord))

    else
        # Recognized call, but not a unary or binary call; copy expression
        fnexpr = :($newvar = $fnold)
        def_fnexpr = fnexpr
        aux_fnexpr = Expr(:nothing)
    end

    return fnexpr, def_fnexpr, aux_fnexpr, def_alloc, aux_alloc
end



"""
`_defs_allocs!(preamble, fnargs, bkkeep, [inloop=false, ex_aux::Expr(:block,)])`

Returns a vector with expressions defining the auxiliary variables
in the preamble, and the declaration of the arrays. This function
may modify `bkkeep.d_indx` if new variables are introduced.
`bkkeep.v_preamb` is for bookkeeping the introduced variables.

"""
function _defs_allocs!(preamble::Expr, fnargs, bkkeep::BookKeeping,
    inloop::Bool, ex_aux::Expr = Expr(:block,))

    # Initializations
    defspreamble = Expr[]
    defs_alloc = Expr[]   # declaration of arrays

    # Local definition for possible args for `@threads` call
    local v_threads_args = (Expr(:., :Threads, QuoteNode(Symbol("@threads"))), Symbol("@threads"))

    for ex in preamble.args
        isa(ex, Expr) || throw(ArgumentError("$ex is not an `Expr`; $(typeof(ex))"))

        # Treat block, for loops, @threads for loops, if separately
        if (ex.head == :block)
            newdefspr = _defs_allocs!(ex, fnargs, bkkeep, inloop, ex_aux)

            append!(defspreamble, newdefspr)

        elseif (ex.head == :for)
            push!(ex_aux.args, ex.args[1])

            newdefspr = _defs_allocs!(ex.args[2], fnargs, bkkeep, true, ex_aux)

            append!(defspreamble, newdefspr)
            pop!(ex_aux.args)

        elseif (ex.head == :macrocall && ex.args[1] in v_threads_args)
            push!(ex_aux.args, ex.args[3].args[1])

            newdefspr = _defs_allocs!(ex.args[3].args[2], fnargs, bkkeep, true, ex_aux)

            append!(defspreamble, newdefspr)
            pop!(ex_aux.args)

        elseif (ex.head == :if)
            for exx in ex.args[2:end]
                newdefspr = _defs_allocs!(exx, fnargs, bkkeep, inloop, ex_aux)

                append!(defspreamble, newdefspr)
            end
            continue

        elseif (ex.head == :local)
            push!(defspreamble, subs(ex, bkkeep.d_indx))
            continue

        elseif (ex.head == :(=))
            # `ex.head` is a :(=) of some kind
            alhs = ex.args[1]
            arhs = ex.args[2]
            arhs = subs(ex.args[2], bkkeep.d_indx) # substitute updated vars in rhs

            # Outside of a loop
            if !inloop
                # `alhs` is declared as an array
                if in(alhs, bkkeep.v_arraydecl) || !in(alhs, bkkeep.v_preamb)
                    push!(defspreamble, subs(ex, bkkeep.d_decl))
                    push!(bkkeep.v_preamb, alhs)
                end

                continue
            end

            # Inside a loop
            if isindexed(alhs)
                # `var1` may be a vector or a matrix; declaring it is subtle
                var1 = alhs.args[1]
                while isa(var1, Expr)
                    var1 = var1.args[1]
                end
                (in(var1, fnargs) || in(var1, bkkeep.v_arraydecl) ||
                    in(alhs, bkkeep.v_preamb)) && continue

                # indices of var1
                d_subs = Dict(veexx.args[1] => veexx.args[2] for veexx in ex_aux.args)
                indx1 = alhs.args[2:end]
                ex_tuple = :( [$(indx1...)] )
                ex_tuple = subs(ex_tuple, d_subs)
                ex_tuple = :( size( $(Expr(:tuple, ex_tuple.args...)) ))
                exx = subs(copy(_DECL_ARRAY),
                    Dict( :__var1 => :($var1), :__var2 => :($ex_tuple)) )

                # Bookkeeping
                push!(defspreamble, exx.args...)
                push!(bkkeep.v_preamb, var1)
                continue

            elseif isindexed(arhs)
                # `alhs` is an aux indexed var, so something in `arhs` is indexed.
                in(alhs, bkkeep.v_preamb) && continue

                vars_indexed = findex(:(_X[_i...]), arhs)

                # NOTE: Uses the size of the var with more indices
                # to define the declaration of the new array.
                ii_indx = argmax( [length(find_indices(aa)[1]) for aa in vars_indexed] )
                var1 = vars_indexed[ii_indx].args[1]
                indx1 = vars_indexed[ii_indx].args[2:end]

                exx_indx = ones(Int, length(indx1))
                exx = subs(copy(_DECL_ARRAY),
                    Dict(:__var1 => :($alhs), :__var2 => :(size($var1))) )

                # Bookkeeping
                inbookkeeping(alhs, bkkeep) && alhs ∈ bkkeep.v_newvars &&
                        deleteat!(bkkeep.v_newvars, findall(x->x==alhs, bkkeep.v_newvars))

                push!(bkkeep.d_indx, alhs => :($alhs[$(indx1...)]) )
                push!(bkkeep.d_decl, alhs => :($alhs[$(exx_indx...)]))
                push!(bkkeep.v_arraydecl, alhs)
                push!(bkkeep.v_preamb, alhs)

                append!(defs_alloc, exx.args)
                continue

            else
                # `alhs` is not indexed nor an aux indexed var
                in(alhs, bkkeep.v_preamb) && continue
                vars_indexed = findex(:(_X[_i...]), arhs)
                if !isempty(vars_indexed)
                    ex = subs(ex, Dict(vv => :(one(S)) for vv in vars_indexed))
                end
                ex = subs(ex, bkkeep.d_decl)
                push!(defspreamble, ex)
                push!(bkkeep.v_preamb, alhs)
                continue
            end

        # # CHECK: Is the following needed?
        # else
        #     # If this is reached, other `ex.head` was encountered (e.g., `.=`);
        #     # we include this in the `defs_tmparrays`
        #     inloop || (push!(defs_tmparrays, subs(ex, bkkeep.d_decl)); continue)
        end

        exx = subs(ex, bkkeep.d_indx)
        inloop || push!(defspreamble, exx)
    end

    # Include allocations first in `defspreamble`
    pushfirst!(defspreamble, defs_alloc...)

    return defspreamble
end


"""
`_recursionloop(fnargs, bkkeep)`

Build the expression for the recursion-loop.

"""
function _recursionloop(fnargs, bkkeep::BookKeeping)
    ll = length(fnargs)

    if ll == 3
        rec_preamb = sanitize( :( $(fnargs[1]).coeffs[2] = $(bkkeep.retvar).coeffs[1] ) )
        rec_fnbody = sanitize( :( $(fnargs[1]).coeffs[ordnext+1] = $(bkkeep.retvar).coeffs[ordnext]/ordnext ) )

    elseif ll == 4
        bkkeep.retvar = fnargs[1]
        rec_preamb = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[2] = $(bkkeep.retvar)[__idx].coeffs[1]
            end))
        rec_fnbody = sanitize(:(
            for __idx in eachindex($(fnargs[2]))
                $(fnargs[2])[__idx].coeffs[ordnext+1] =
                    $(bkkeep.retvar)[__idx].coeffs[ordnext]/ordnext
            end))

    else
        throw(ArgumentError(
        "Wrong number of arguments in the definition of the function $fn"))
    end

    return rec_preamb, rec_fnbody
end


"""
`_split_arraydecl!(bkkeep)`

Split bkkeep.v_arraydecl in the vector (bkkeep.v_array1), matrix (bkkeep.v_array2), etc,
to properly construct the `RetAlloc` variable.
"""
function _split_arraydecl!(bkkeep::BookKeeping)
    for s in bkkeep.v_arraydecl
        for v in values(bkkeep.d_indx)
            if v.head == :ref && v.args[1] == s
                if length(v.args) == 2
                    push!(bkkeep.v_array1, s)
                    break
                elseif length(v.args) == 3
                    push!(bkkeep.v_array2, s)
                    break
                elseif length(v.args) == 4
                    push!(bkkeep.v_array3, s)
                    break
                elseif length(v.args) == 5
                    push!(bkkeep.v_array4, s)
                    break
                else
                    error("Error: `@taylorize` allows only to parse up tp 5-index arrays")
                end
            end
        end
    end
    return nothing
end


"""
`_allocated_defs!(new_jetcoeffs, bkkeep)`

Add allocated variable definitions to `new_jetcoeffs`, to make it more human readable.
"""
function _allocated_defs!(new_jetcoeffs::Expr, bkkeep::BookKeeping)
    tmp_defs = [popfirst!(new_jetcoeffs.args[2].args)]
    @inbounds for (ind, vnew) in enumerate(bkkeep.v_newvars)
        push!(tmp_defs, :($(vnew) = __ralloc.v0[$(ind)]))
    end
    @inbounds for (ind, vnew) in enumerate(bkkeep.v_array1)
        push!(tmp_defs, :($(vnew) = __ralloc.v1[$(ind)]))
    end
    @inbounds for (ind, vnew) in enumerate(bkkeep.v_array2)
        push!(tmp_defs, :($(vnew) = __ralloc.v2[$(ind)]))
    end
    @inbounds for (ind, vnew) in enumerate(bkkeep.v_array3)
        push!(tmp_defs, :($(vnew) = __ralloc.v3[$(ind)]))
    end
    @inbounds for (ind, vnew) in enumerate(bkkeep.v_array4)
        push!(tmp_defs, :($(vnew) = __ralloc.v4[$(ind)]))
    end
    prepend!(new_jetcoeffs.args[2].args, tmp_defs)
    return nothing
end


"""
`_returned_expr(bkkeep)`

Constructs the expression to be returned by `TaylorIntegration._allocate_jetcoeffs!`
"""
function _returned_expr(bkkeep::BookKeeping)
    if isempty(bkkeep.v_newvars)
        retv0 = :(Taylor1{_S}[])
    else
        retv0 = :([$(bkkeep.v_newvars...),])
    end
    if isempty(bkkeep.v_array1)
        retv1 = :([Array{Taylor1{_S},1}(undef, 0),])
    else
        retv1 = :([$(bkkeep.v_array1...),])
    end
    if isempty(bkkeep.v_array2)
        retv2 = :([Array{Taylor1{_S},2}(undef, 0, 0),])
    else
        retv2 = :([$(bkkeep.v_array2...),])
    end
    if isempty(bkkeep.v_array3)
        retv3 = :([Array{Taylor1{_S},3}(undef, 0, 0, 0),])
    else
        retv3 = :([$(bkkeep.v_array3...),])
    end
    if isempty(bkkeep.v_array4)
        retv4 = :([Array{Taylor1{_S},4}(undef, 0, 0, 0, 0),])
    else
        retv4 = :([$(bkkeep.v_array4...),])
    end

    return :(return TaylorIntegration.RetAlloc{Taylor1{_S}}(
        $(retv0), $(retv1), $(retv2), $(retv3), $(retv4)))
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
    nex1, nex2 = _make_parsed_jetcoeffs(ex)
    esc(quote
        $(ex)   # evals to calling scope the passed function
        $(nex1) # evals the new method of `jetcoeffs!`
        $(nex2) # evals the new method of `_allocate_jetcoeffs`
    end)
end
