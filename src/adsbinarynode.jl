# This file is part of the TaylorIntegration.jl package; MIT licensed

using StaticArrays: SVector
using AbstractTrees: StoredParents, HasNodeType, TreeIterator, PreOrderDFS
import AbstractTrees

if !isdefined(Base, :isnothing)        # Julia 1.0 support
    using AbstractTrees: isnothing
end

export ADSDomain, ADSBinaryNode

"""
    ADSDomain{N, T <: Real}

`N` dimensional box used in Automatic Domain Splitting to represent the
domain of `N` jet transport variables.

# Fields

- `lo::SVector{N, T}`: lower bounds.
- `hi::SVector{N, T}`: upper bounds.
"""
struct ADSDomain{N, T <: Real}
    lo::SVector{N, T}
    hi::SVector{N, T}
    # Inner constructor
    function ADSDomain{N, T}(lo::SVector{N, T}, hi::SVector{N, T}) where {N, T <: Real}
        @assert all(hi .> lo) "Each upper bound must be greater than the respective lower bound"
        new{N, T}(lo, hi)
    end
end

# Outer constructors
ADSDomain(lo::SVector{N, T}, hi::SVector{N, T}) where {N, T <: Real} = ADSDomain{N, T}(lo, hi)

ADSDomain() = ADSDomain{1, Float64}(SVector{1, Float64}(zero(Float64)), SVector{1, Float64}(one(Float64)))

function ADSDomain(x::Tuple{T, T}...) where {T <: Real}
    N = length(x)
    return ADSDomain{N, T}(
        SVector{N, T}(x[i][1] for i in eachindex(x)),
        SVector{N, T}(x[i][2] for i in eachindex(x))
    )
end

# Print method for ADSDomain
# Example:
# [-1.0, 1.0]×[-1.0, 1.0]
function show(io::IO, s::ADSDomain{N, T}) where {N, T <: Real}
    x = Vector{String}(undef, N)
    for i in eachindex(x)
        x[i] = string(SVector{2, T}(s.lo[i], s.hi[i]))
    end
    print(io, join(x, "×"))
end

# Return the diameter of each direction in s
widths(s::ADSDomain{N, T}) where {N, T <: Real} = s.hi - s.lo
# Return the lower bounds in s
infima(s::ADSDomain{N, T}) where {N, T <: Real} = s.lo
# Return the upper bounds in s
suprema(s::ADSDomain{N, T}) where {N, T <: Real} = s.hi
# Split s in half along direction i
function Base.split(s::ADSDomain{N, T}, i::Int) where {N, T <: Real}
    @assert 1 <= i <= N
    mid = (s.lo[i] + s.hi[i])/2
    a = ADSDomain{N, T}(
        s.lo,
        SVector{N, T}(i == j ? mid : s.hi[j] for j in 1:N)
    )
    b = ADSDomain{N, T}(
        SVector{N, T}(i == j ? mid : s.lo[j] for j in 1:N),
        s.hi
    )
    return a, b
end

function Base.in(x::AbstractVector{T}, s::ADSDomain{N, T}) where {N, T <: Real}
    @assert length(x) == N "x must be of length $N"
    mask = SVector{N, Bool}(s.lo[i] <= x[i] <= s.hi[i] for i in 1:N)
    return all(mask)
end

"""
    ADSBinaryNode{N, M, T <: Real}

Automatic Domain Splitting binary tree with `N` jet transport variables
and `M` degrees of freedom.

# Fields

- `s::ADSDomain{N, T}`: domain of jet transport variables.
- `t::T`: time.
- `x::SVector{M, TaylorN{T}}`: state vector at time `t`.
- `p::SVector{M, Taylor1{TaylorN{T}}}`: polynomial dependence of state vector wrt time.
- `depth::Int`: node depth.
- `parent::Union{Nothing, ADSBinaryNode{N, M, T}}`: parent node.
- `left::Union{Nothing, ADSBinaryNode{N, M, T}}`: left child.
- `right::Union{Nothing, ADSBinaryNode{N, M, T}}`: right child.

!!! reference
    This implementation is based upon the **binarytree.jl** example at
    https://github.com/JuliaCollections/AbstractTrees.jl/blob/master/test/examples/binarytree.jl
"""
mutable struct ADSBinaryNode{N, M, T <: Real}
    s::ADSDomain{N, T}
    t::T
    x::SVector{M, TaylorN{T}}
    p::SVector{M, Taylor1{TaylorN{T}}}
    depth::Int
    parent::Union{Nothing, ADSBinaryNode{N, M, T}}
    left::Union{Nothing, ADSBinaryNode{N, M, T}}
    right::Union{Nothing, ADSBinaryNode{N, M, T}}
    # Inner constructor (root node by default)
    function ADSBinaryNode{N, M, T}(
        s::ADSDomain{N, T}, t::T, x::SVector{M, TaylorN{T}},
        p::SVector{M, Taylor1{TaylorN{T}}}, depth::Int = 0,
        parent = nothing, l = nothing, r = nothing
        ) where {N, M, T <: Real}
        new{N, M, T}(s, t, x, p, depth, parent, l, r)
    end
end

# Outer constructor (root node)
function ADSBinaryNode(
    s::ADSDomain{N, T}, t::T, x::SVector{M, TaylorN{T}},
    p::SVector{M, Taylor1{TaylorN{T}}}
    ) where {N, M, T <: Real}
    return ADSBinaryNode{N, M, T}(s, t, x, p)
end

# Print method for ADSBinaryNode
# Example:
# x: [1.0, 0.0, 0.0, 1.224744871391589]
# s: [-1.0, 1.0]×[-1.0, 1.0]
# t: 0.0
function show(io::IO, n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}
    print(io, "x: ", constant_term.(n.x), "\ns: ", n.s, "\nt: ", n.t)
end

# Left (right) child constructors
function leftchild!(
    parent::ADSBinaryNode{N, M, T}, s::ADSDomain{N, T}, t::T,
    x::SVector{M, TaylorN{T}}, p::SVector{M, Taylor1{TaylorN{T}}}
    ) where {N, M, T <: Real}
    isnothing(parent.left) || error("Left child is already assigned")
    node = ADSBinaryNode{N, M, T}(s, t, x, p, parent.depth + 1, parent)
    parent.left = node
end

function leftchild!(parent::ADSBinaryNode{N, M, T}, node::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}
    # set `node` as left child of `parent`
    parent.left = node
end

function rightchild!(
    parent::ADSBinaryNode{N, M, T}, s::ADSDomain{N, T}, t::T,
    x::SVector{M, TaylorN{T}}, p::SVector{M, Taylor1{TaylorN{T}}}
    ) where {N, M, T <: Real}
    isnothing(parent.right) || error("Right child is already assigned")
    node = ADSBinaryNode{N, M, T}(s, t, x, p, parent.depth + 1, parent)
    parent.right = node
end

function rightchild!(parent::ADSBinaryNode{N, M, T}, node::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}
    # set `node` as right child of `parent`
    parent.right = node
end

# AbstractTrees interface
function AbstractTrees.children(node::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}
    if isnothing(node.left) && isnothing(node.right)
        ()
    elseif isnothing(node.left) && !isnothing(node.right)
        (node.right,)
    elseif !isnothing(node.left) && isnothing(node.right)
        (node.left,)
    else
        (node.left, node.right)
    end
end

AbstractTrees.printnode(io::IO, n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real} = print(io, "t: ", n.t)

AbstractTrees.nodevalue(n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real} = (n.s, n.t, n.x, n.p)

AbstractTrees.ParentLinks(::Type{<:ADSBinaryNode{N, M, T}}) where {N, M, T <: Real} = StoredParents()

AbstractTrees.parent(n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real} = n.parent

AbstractTrees.NodeType(::Type{<:ADSBinaryNode{N, M, T}}) where {N, M, T <: Real} = HasNodeType()
AbstractTrees.nodetype(::Type{<:ADSBinaryNode{N, M, T}}) where {N, M, T <: Real} = ADSBinaryNode{N, M, T}

# For TreeIterator
Base.IteratorEltype(::Type{<:TreeIterator{ADSBinaryNode{N, M, T}}}) where {N, M, T <: Real} = HasEltype()
Base.eltype(::Type{<:TreeIterator{ADSBinaryNode{N, M, T}}}) where {N, M, T <: Real} = ADSBinaryNode{N, M, T}

"""
    countnodes(n::ADSBinaryNode{N, M, T}, k::Int) where {N, M, T <: Real}

Return the number of nodes at depth `k` starting from root node `n`.
"""
function countnodes(n::ADSBinaryNode{N, M, T}, k::Int) where {N, M, T <: Real}
    @assert k >= 0 "k must be greater or equal to 0"
    if iszero(k)
        return 1
    else
        return countnodes(n.left, k-1) + countnodes(n.right, k-1)
    end
end

countnodes(n::Nothing, k::Int) = 0

"""
    countnodes(n::ADSBinaryNode{N, M, T}, t::T) where {N, M, T <: Real}

Return the number of nodes at time `t` starting from root node `n`.
"""
function countnodes(n::ADSBinaryNode{N, M, T}, t::T) where {N, M, T <: Real}
    if !isnothing(n.left) && (n.t < n.left.t && t < n.t) || (n.t > n.left.t && t > n.t)
        # t is outside time range
        return 0
    else
        # Count nodes at time t (recursively)
        return _countnodes(n, t)
    end
end

function _countnodes(n::ADSBinaryNode{N, M, T}, t::T) where {N, M, T <: Real}
    # Count nodes at time t (recursively)
    if !isnothing(n.parent) && (abs(n.parent.t) <= abs(t) < abs(n.t)) || (t == n.t && isnothing(n.left))
        return 1
    else
        return _countnodes(n.left, t) + _countnodes(n.right, t)
    end
end

_countnodes(n::Nothing, t::T) where {T <: Real} = 0

"""
    timeshift!(n::ADSBinaryNode{N, M, T}, dt::T) where {N, M, T <: Real}

Shift the time of every node in `n` by `dt`.
"""
function timeshift!(n::ADSBinaryNode{N, M, T}, dt::T) where {N, M, T <: Real}
    for node in PreOrderDFS(n)
        node.t += dt
    end
end

"""
    timesvector(n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}

Return the vector of times starting from root node `n`.
"""
function timesvector(n::ADSBinaryNode{N, M, T}) where {N, M, T <: Real}
    # Allocate set of times
    ts = Set{T}()
    # Search times (recursively)
    timesvector!(n, ts)
    # Set{T} to Vector{T}
    times = collect(ts)
    # Forward tree
    if sign(times[findmax(abs, times)[2]]) == 1
        return sort!(times)
    # Backward tree
    else
        return sort!(times, rev = true)
    end
end

function timesvector!(n::ADSBinaryNode{N, M, T}, ts::Set{T}) where {N, M, T <: Real}
    push!(ts, n.t)
    timesvector!(n.left, ts)
    timesvector!(n.right, ts)
    nothing
end

timesvector!(n::Nothing, ts::Set{T}) where {T <: Real} = nothing

# Auxiliary evaluation methods
function _adseval(p::SVector{M, Taylor1{TaylorN{T}}}, dt::U) where {M, T <: Real, U <: Number}
    return SVector{M, TaylorN{T}}(p[i](dt) for i in 1:M)
end

"""
    evaltree(n::ADSBinaryNode{N, M, T}, t::U) where {N, M, T <: Real, U <: Number}

Evaluate binary tree `n` at time `t`.
"""
function evaltree(n::ADSBinaryNode{N, M, T}, t::U) where {N, M, T <: Real, U <: Number}
    # t is outside time range
    if !isnothing(n.left) && (n.t < n.left.t && t < n.t) || (n.t > n.left.t && t > n.t)
        return Vector{ADSDomain{N, T}}(undef, 0), Matrix{TaylorN{T}}(undef, 0, 0)
    end
    # Allocate set of nodes
    ns = Set{ADSBinaryNode{N, M, T}}()
    # Search nodes at time  t (recursively)
    evaltree!(n, t, ns)
    # Set{ADSBinaryNode{N, M, T}} to Vector{ADSBinaryNode{N, M, T}}
    nodes = collect(ns)
    # Sort nodes by lowest corner of domain
    sort!(nodes, by = x -> x.s.lo)
    # Number of nodes at time t
    L = length(nodes)
    # There are 0 nodes at time t
    if iszero(L)
        s = Vector{ADSDomain{N, T}}(undef, 0)
        x = Matrix{TaylorN{T}}(undef, 0, 0)
    # Evaluate polynomials at time t
    else
        # Domain of each node
        s = Vector{ADSDomain{N, T}}(undef, L)
        # State vector of each node
        x = Matrix{TaylorN{T}}(undef, M, L)

        for i in eachindex(nodes)
            dt = t - nodes[i].parent.t
            s[i] = nodes[i].s
            x[:, i] .= _adseval(nodes[i].p, dt)
        end
    end

    return s, x
end

function evaltree!(n::ADSBinaryNode{N, M, T}, t::U,
                   ns::Set{ADSBinaryNode{N, M, T}}) where {N, M, T <: Real, U <: Number}
    if !isnothing(n.parent) && (abs(n.parent.t) <= abs(t) < abs(n.t)) || (t == n.t && isnothing(n.left))
            push!(ns, n)
    else
        evaltree!(n.left, t, ns)
        evaltree!(n.right, t, ns)
    end
    nothing
end

evaltree!(n::Nothing, t::U, ns::Set{ADSBinaryNode{N, M, T}}) where {N, M, T <: Real, U <: Number} = nothing

# Function-like callability method
(n::ADSBinaryNode{N, M, T})(t::U) where {N, M, T <: Real, U <: Number} = evaltree(n, t)

"""
    evaltree(n::ADSBinaryNode{N, M, T}, t::U,
             s::AbstractVector{T}) where {N, M, T <: Real, U <: Number}

Evaluate binary tree `n` at time `t` and domain point `s`.
"""
function evaltree(n::ADSBinaryNode{N, M, T}, t::U,
                  s::AbstractVector{T}) where {N, M, T <: Real, U <: Number}
    # Check length(s)
    @assert length(s) == N "s must be of length $N"
    # t is outside time range or s is outside domain
    if !(s in n.s) || !isnothing(n.left) && (n.t < n.left.t && t < n.t) ||
        (n.t > n.left.t && t > n.t)
        return Vector{T}(undef, 0)
    end
    # Allocate set of nodes
    ns = Set{ADSBinaryNode{N, M, T}}()
    # Search nodes at time t and domain s (recursively)
    evaltree!(n, t, s, ns)
    # Set{ADSBinaryNode{N, M, T}} to Vector{ADSBinaryNode{N, M, T}}
    nodes = collect(ns)
    # Number of nodes at time t and domain s
    L = length(nodes)
    # There are 0 nodes at time t and domain s
    if iszero(L)
        x = Vector{T}(undef, 0)
    # Evaluate polynomials at time t and domain s
    else
        # Choose the first node
        node = nodes[1]
        # Time delta
        dt = t - node.parent.t
        # Evaluate node polynomial at dt
        p = _adseval(node.p, dt)
        # Root and local domain
        sup = getroot(n).s
        loc = node.s
        # Linear transformation
        ms = widths(sup) ./ widths(loc)
        ks = infima(sup) - infima(loc) .* ms
        _s_ = ms .* s .+ ks
        # Eval p at transformed point
        x = map(y -> y(_s_), p)
    end

    return x
end

function evaltree!(n::ADSBinaryNode{N, M, T}, t::U, s::AbstractVector{T},
                   ns::Set{ADSBinaryNode{N, M, T}}) where {N, M, T <: Real, U <: Number}
    !(s in n.s) && return nothing
    if !isnothing(n.parent) && (abs(n.parent.t) <= abs(t) < abs(n.t)) ||
        (t == n.t && isnothing(n.left))
        push!(ns, n)
    else
        evaltree!(n.left, t, s, ns)
        evaltree!(n.right, t, s, ns)
    end
    nothing
end

evaltree!(n::Nothing, t::U, s::AbstractVector{T},
          ns::Set{ADSBinaryNode{N, M, T}}) where {N, M, T <: Real, U <: Number} = nothing

# Function-like callability method
function (n::ADSBinaryNode{N, M, T})(t::U, s::AbstractVector{T}) where {N, M, T <: Real, U <: Number}
    return evaltree(n, t, s)
end