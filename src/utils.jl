# Convenience const. 
const AbsTen{N₁,N₂,S,T} = AbstractTensorMap{T,S,N₁,N₂}
const TenAbs{N₂,N₁,S,T} = AbstractTensorMap{T,S,N₁,N₂}

const _DIR_TO_DOMIND = (east=1, south=2, west=3, north=4)

# Output the permutation of that results in the k'th element of p moved to the
# n'th position 
function nswap(k::Int, n::Int, p::NTuple{N}) where {N}
    if k == n
        return p
    elseif n > k
        return (p[1:(k - 1)]..., p[(k + 1):n]..., p[k]..., p[(n + 1):N]...)::typeof(p)
    else
        return (p[1:(n - 1)]..., p[k], p[n:(k - 1)]..., p[(k + 1):N]...)::typeof(p)
    end
end

space2tuple(space::ProductSpace{S,M}) where {S,M} = NTuple{M,S}(space)
space2tuple(space) = space2tuple(convert(ProductSpace, space))

mapadjoint(space) = prod(map(adjoint, space2tuple((space))))

@doc raw"""
    swap(k::Int, l::Int, p::NTuple)

Given `p`, return the `Tuple` with elements `k` and `l` swapped. Note, `k` and `l` must both 
be less than or equal to `length(p)`. 
"""
function swap(k::Int, l::Int, p::NTuple{N}) where {N}
    if !(k <= N && l <= N)
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    v = Vector{Int}([p...])
    v[k] = p[l]
    v[l] = p[k]
    return typeof(p)(v)
end

delta(x, y) = ==(x, y)
function delta(z...)
    for i in z
        !delta(z[1], i) && return false
    end
    return true
end

# type stable
function delta(T::Type{<:Number}, dim1::Int, dim2::Int, dims::Vararg{Int,N}) where {N}
    a = Array{T}(undef, dim1, dim2, dims...)
    for (i, _) in pairs(a)
        a[i] = T(delta(i))
    end
    return a
end

delta(I::CartesianIndex) = delta(Tuple(I)...)

function delta(T::Type{<:Number}, cod::VectorSpace, dom::VectorSpace)
    return TensorMap(s -> delta(T, s...), cod, dom)
end

function slice(ind::NTuple{N,Int}, sp::ProductSpace{S,N₁}) where {N,N₁,S<:IndexSpace}
    return *(one(S), one(S), (sp[i] for i in ind)...)
end

# Alt permute() 
function _permute(t::AbsTen{N,M,<:IndexSpace}, p::Tuple) where {N,M}
    if !(length(p) == N + M)
        throw(DimensionMismatch())
    else
        return permute(t, (p[1:N], p[(N + 1):(N + M)]))
    end
end

# Swap codomain and domain (Base.transpose reverses indices)
function _transpose(tsrc::AbsTen{N,M}) where {N,M}
    return permute(
        tsrc, (Tuple((N + 1):(N + M))::NTuple{M::Int}, Tuple(1:N)::NTuple{N,Int})
    )
end

function permutecod(t::AbsTen{N,M}, p::NTuple{N,Int}) where {N,M}
    return permute(t, (p, Tuple((N + 1):(M + N))::NTuple{M,Int}))
end

function permutedom(t::AbsTen{N,M}, p::NTuple{M,Int}) where {N,M}
    p_dom = p .+ N
    return permute(t, (Tuple(1:N)::NTuple{N,Int}, p_dom))
end
function permutecod!(tdst, tsrc::AbsTen{N,M}, p::NTuple{N,Int}) where {N,M}
    return permute!(tdst, tsrc, (p, Tuple((N + 1):(M + N))::NTuple{M,Int}))
end

function permutedom!(tdst, tsrc::AbsTen{N,M}, p::NTuple{M,Int}) where {N,M}
    p_dom = p .+ N
    return permute!(tdst, tsrc, (Tuple(1:N)::NTuple{N,Int}, p_dom))
end

function _leftorth(t::AbsTen{N,2}) where {N}
    Q, R = leftorth(t, (tuple(1:N..., N + 2)::NTuple{N + 1,Int}, (N + 1,)))
    t_out = permute(Q, (Tuple(1:N)::NTuple{N,Int}, (N + 2, N + 1)))
    r_out = permute(R, ((), (2, 1)))
    return t_out, r_out
end

function _rightorth(t::AbsTen{N,2}) where {N}
    L, Q = rightorth(t, ((N + 2,), tuple(1:N..., N + 1)::NTuple{N + 1,Int}))
    t_out = permute(Q, (Tuple(2:(N + 1))::NTuple{N,Int}, (N + 2, 1)))
    l_out = permute(L, ((), (2, 1)))
    return l_out, t_out
end

# Convert a bond dimension to an index space of type SType
function dimtospace(SType::Type{<:IndexSpace}, D::Int)
    S = oneunit(SType)
    if D == 1
        return S::SType
    else
        return ⊕(S^D...)::SType
    end
end

function tensormap(
    arr::AbstractArray, p1::NTuple{N₁,<:Integer}, p2::NTuple{N₂,<:Integer}
) where {N₁,N₂}
    arr_p = permutedims(arr, (p1..., p2...))
    space = map(x -> ℂ^2, size(arr_p))

    nil = one(ComplexSpace)

    tmult = x -> splat(*)((nil, nil, x...))

    cod = tmult(space[1:N₁])
    dom = tmult(space[(N₁ + 1):(N₁ + N₂)])

    return TensorMap(arr_p, cod, dom)
end

deepcopy!(dst::AbstractArray, src::AbstractArray) = copy!(dst, src)

function isinitialised(arr::AbstractArray)
    local rv
    try
        rv = all(isinitialised.(arr, LinearIndices(arr)))
    catch e
        if isa(e, UndefRefError)
            rv = false
        else
            throw(e)
        end
    end
    return rv
end
function isinitialised(arr::AbstractTensorMap, i)
    return isassigned(arr.data, i)
end
isinitialised(arr, i) = isassigned(arr, i)

# n * π/2
rotate(ten::TenAbs{4}, n) = permutedom(ten, tcircshift((1, 2, 3, 4), n))

function tcircshift(tup::NTuple{N,T}, i::Int) where {N,T}
    tup_v = [tup...]
    rv = Tuple(circshift(tup_v, i))::NTuple{N,T}
    return rv
end

function swap(s::TensorSpace{S}) where {S}
    ps = convert(ProductSpace, s)
    return *(one(S), adjoint.(ps)...)::typeof(ps)
end

function forcehermitian(args...)
    T = promote_type(map(scalartype, args)...)
    if T <: Real
        return true
    else
        return false
    end
end

function conjugate(t::AbstractTensorMap)
    return TensorMap(conj(t.data), codomain(t), domain(t))
end

const cj = conjugate

function get_embedding_isometry(bond, chi)
    if bond ≾ chi
        iso = transpose(isometry(chi', bond'))
    else
        iso = isometry(bond, chi)
    end
end

get_removal_isometry(bond) = get_embedding_isometry(bond, one(bond))

@generated function deepcopy!(ten_dst::T, ten_src::T) where {T}
    assignments = [:(deepcopy!(ten_dst.$name, ten_src.$name)) for name in fieldnames(T)]
    quote
        $(assignments...)
    end
end
