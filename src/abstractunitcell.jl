abstract type AbstractUnitCellGeometry end
abstract type AbstractUnitCell{G<:AbstractUnitCellGeometry,ElType,A} <:
              AbstractMatrix{ElType} end

basedata(arr::CircularArray) = arr
basedata(arr::AbstractUnitCell) = basedata(getdata(arr))

const AbUnCe{T,G,A} = AbstractUnitCell{G,T,A}

tensortype(val) = tensortype(typeof(val))
tensortype(args::Type) = throw(MethodError(tensortype, (args,)))
tensortype(T::Type{<:AbstractTensorMap}) = T
tensortype(::Type{<:AbUnCe{T}}) where {T} = tensortype(T)

"""
$(TYPEDEF)

Singleton type representing a standard square lattice geometry with coordination number 4.
"""
struct Square <: AbstractUnitCellGeometry end

"""
$(TYPEDEF)

Singleton type representing a square lattice with reverse cyclic symmetry. For (2,2) unit
cell sizes, this defines a checkerboard lattice geometry.

# Examples

```jldoctest
julia> UnitCell{SquareSymmetric}([1 2 3; 2 3 1; 3 1 2])
3×3 UnitCell{SquareSymmetric, Int64, Matrix{Int64}}:
 1  2  3
 2  3  1
 3  1  2

julia> UnitCell{SquareSymmetric}([1 2; 3 4])
ERROR: ArgumentError: data does not have the required cyclic symmetry for this lattice geometry.
```
"""
struct SquareSymmetric <: AbstractUnitCellGeometry end

"""
$(TYPEDEF)

Container `AbstractMatrix` type with periodic boundary conditions, and a lattice geometry 
specific by type parameter `G`.
"""
struct UnitCell{G<:AbstractUnitCellGeometry,ElType,A} <: AbstractUnitCell{G,ElType,A}
    data::CircularArray{ElType,2,A}
    UnitCell{G,T,A}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
    UnitCell{G}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
    function UnitCell{SquareSymmetric}(data::CircularArray{T,2,A}) where {T,A}
        nx, ny = size(data)
        if ny > 1
            if nx == ny
                for i in 2:ny
                    if !all(data[:, 1] .== circshift(data[:, i], (i - 1)))
                        throw(
                            ArgumentError(
                                "data does not have the required cyclic symmetry for this lattice geometry.",
                            ),
                        )
                    end
                end
            else
                if nx == 1
                    data = permutedims(data)
                else
                    throw(ArgumentError("data not square."))
                end
            end
        end
        return new{SquareSymmetric,T,A}(data[:, 1:1])
    end
    function UnitCell{SquareSymmetric}(
        ::UndefInitializer, protomat::CircularArray{T,2,A}
    ) where {T,A}
        n, _ = size(protomat)
        data = similar(protomat, (n, 1))
        return new{SquareSymmetric,T,A}(data)
    end
end

UnitCell(uc::UnitCell) = uc
UnitCell{SquareSymmetric}(vec::AbstractVector) = UnitCell{SquareSymmetric}(hcat(vec))

getdata(uc::UnitCell{G,T,A}) where {G,T,A} = uc.data::CircularArray{T,2,A}

datatype(::Type{<:AbstractUnitCell{G,ElType,A}}) where {G,ElType,A} = A
datatype(U::Type) = U

"""
    $(FUNCTIONNAME)(::AbstractUnitCell) -> Type{G}
    $(FUNCTIONNAME)(::Type{<:AbstractUnitCell}) -> Type{G}

Return the geometry type `G` of a unit cell.
"""
geometrytype(uc::AbstractUnitCell{G}) where {G} = geometrytype(typeof(uc))
geometrytype(::Type{<:AbstractUnitCell{G}}) where {G} = G

## ABSTRACT ARRAY

Base.size(uc::AbstractUnitCell) = size(getdata(uc))
function Base.size(uc::AbstractUnitCell{SquareSymmetric})
    nx = size(getdata(uc), 1)
    return (nx, nx)
end

function Base.getindex(uc::AbstractUnitCell{G}, inds...) where {G}
    return _getindex(G, uc, to_indices(uc, inds)...)
end

_getindex(::Type{<:AbstractUnitCellGeometry}, uc, inds...) = getindex(basedata(uc), inds...)

Base.setindex!(uc::AbstractUnitCell, v, i...) = setindex!(getdata(uc), v, i...)
function Base.setindex!(uc::AbstractUnitCell{SquareSymmetric}, v, i1, i2)
    if !(i1 isa Integer)
        throw(
            ArgumentError(
                "setting elemets of `UnitCell{SquareSymmetric}` requires i1::Int or i2::Int (or both)",
            ),
        )
    end
    setindex!(uc, v, i2, i1)
    return uc
end

function Base.setindex!(uc::AbstractUnitCell, val, i)
    # Single argument method defers to the two argument method
    inds = CircularArray(CartesianIndices(uc))

    cartind = inds[i]

    return setindex!(uc, val, to_indices(uc, (cartind,))...)
end
function Base.setindex!(uc::AbstractUnitCell{SquareSymmetric}, v, i1, i2::Int)
    setindex!(basedata(uc), v, i1 .+ (i2 - 1), 1)
    return uc
end

# SquareSymmetric
function _getindex(G::Type{SquareSymmetric}, uc::AbstractUnitCell, i)
    # Single argument method defers to the two argument method
    inds = CircularArray(CartesianIndices(uc))

    cartind = inds[i]

    return _getindex(G, uc, to_indices(uc, (cartind,))...)
end
function _getindex(
    G::Type{SquareSymmetric}, uc::AbstractUnitCell, i1::Union{UnitRange,Base.Slice}, i2::Int
)
    return _getindex(G::Type{SquareSymmetric}, uc::AbstractUnitCell, i2, i1)
end
function _getindex(G::Type{SquareSymmetric}, uc::AbstractUnitCell, i1, i2)
    if isempty(i1)
        if isempty(i2)
            return similar(basedata(uc), 0, 0)
        end
        return permutedims(_getindex(G, uc, i2, i1))
    else
        d = length(i2) == 1 ? 1 : 2
        rv = mapreduce((x...) -> cat(x...; dims=d), i1) do i
            return getindex(circshift(basedata(uc), -(i - 1)), i2)
        end
        return maybe2circular(eltype(uc), rv)
    end
end

maybe2circular(::Type{T}, val::T) where {T} = val
maybe2circular(::Type{T}, arr::CircularArray{T}) where {T} = arr
maybe2circular(::Type{T}, arr::AbstractArray{T}) where {T} = CircularArray(arr)

## 

## SIMILAR

Base.similar(uc::UnitCell{G,T}) where {G,T} = similar(uc, T)
Base.similar(uc::UnitCell, ::Type{S}) where {S} = similar(uc, S, size(uc))
Base.similar(uc::UnitCell{G,T}, dims::Dims) where {T,G} = similar(uc, T, dims)
function Base.similar(uc::UnitCell{G,T}, ::Type{S}, dims::Dims) where {T,S,G}
    return UnitCell{G}(similar(getdata(uc), S, dims))
end
function Base.similar(uc::UnitCell{SquareSymmetric,T}, ::Type{S}, dims::Dims) where {T,S}
    return UnitCell{SquareSymmetric}(undef, similar(getdata(uc), S, dims))
end

function UnitCell{SquareSymmetric}(::UndefInitializer, protomat::AbstractMatrix)
    return UnitCell{SquareSymmetric}(undef, tocircular(protomat))
end
##

## BROADCASTING

# Pass in the `BroadcastStyle` such that we can get compute the winning broadcast style
# of the underlying abstract matrix.

abstract type AbstractUnitCellStyle{G,A<:Base.BroadcastStyle} <:
              Broadcast.AbstractArrayStyle{2} end

struct UnitCellStyle{G,A} <: AbstractUnitCellStyle{G,A} end
struct UnitCellUnsafeStyle{G,A} <: AbstractUnitCellStyle{G,A} end

(T::Type{<:AbstractUnitCellStyle})(::Val{2}) = T()
(T::Type{<:AbstractUnitCellStyle})(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

function Broadcast.BroadcastStyle(::Type{UnitCell{G,ElType,A}}) where {G,ElType,A}
    return UnitCellStyle{G,typeof(Broadcast.BroadcastStyle(A))}()
end
function Broadcast.BroadcastStyle(::Type{<:AbstractUnitCell{G,ElType,A}}) where {G,ElType,A}
    return UnitCellStyle{G,typeof(Broadcast.BroadcastStyle(A))}()
end

function Broadcast.BroadcastStyle(
    ::UnitCellStyle{G,A}, ::Broadcast.ArrayStyle{<:CircularArray{ElType,2,B}}
) where {G,ElType,A,B}
    AB = Broadcast.BroadcastStyle(A(), Broadcast.BroadcastStyle(B))
    return UnitCellStyle{G,typeof(AB)}()
end

function Broadcast.BroadcastStyle(
    ::UnitCellStyle{SquareSymmetric,A}, ::B
) where {A,B<:Broadcast.DefaultArrayStyle{2}}
    AB = Broadcast.BroadcastStyle(A(), B())
    return UnitCellUnsafeStyle{Square,typeof(AB)}()
end
function Broadcast.BroadcastStyle(
    ::UnitCellStyle{G,A}, ::B
) where {G,A,B<:Broadcast.DefaultArrayStyle{2}}
    AB = Broadcast.BroadcastStyle(A(), B())
    return UnitCellStyle{G,typeof(AB)}()
end

function Broadcast.BroadcastStyle(
    ::UnitCellStyle{G1,A}, ::UnitCellStyle{G2,B}
) where {A,B,G1,G2}
    AB = Broadcast.BroadcastStyle(A(), B())
    G = promote_geometry(G1, G2)
    return UnitCellStyle{G,typeof(AB)}()
end

promote_geometry(::Type{T1}, ::Type{T2}) where {T1,T2} = promote_geometry(T2, T1)
promote_geometry(::Type{Square}, ::Type{SquareSymmetric}) = Square

function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return UnitCell{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end

function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellStyle{SquareSymmetric,A}}, ::Type{ElType}
) where {A,ElType}
    return UnitCell{SquareSymmetric}(
        undef, similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType)
    )
end

## CONSTRUCTORS

tocircular(data::AbstractMatrix) = CircularArray(data)
tocircular(data::CircularArray) = data
tocircular(data::AbstractUnitCell) = getdata(data)

# Make circular
"""
    $(FUNCTIONNAME)(data::AbstractMatrix) -> UnitCell{Square}
    $(FUNCTIONNAME){G}(data::AbstractMatrix) -> UnitCell{G}

Construct a unit cell with elements given by the elements of `data`.
"""
UnitCell(data) = UnitCell{Square}(data)
UnitCell{G}(data) where {G} = UnitCell{G}(tocircular(data))

## VIEW (Probably deprec)

@inline function Base.view(uc::AbstractUnitCell, i1, i2, inds...)
    # @debug "Using UnitCell view..."
    new_inds = (int_to_range(i1, i2)..., inds...)
    return UnitCell(view(getdata(uc), new_inds...))
end

const SubUnitCell{G,T,A<:SubArray} = UnitCell{G,T,A}

_unitrange(i::Int) = UnitRange(i, i)
_unitrange(x) = x

int_to_range(i1, i2) = (_unitrange(i1), _unitrange(i2))

## UTILS

size_allequal(ucs...) = allequal(size.(ucs...))
function check_size_allequal(ucs...)
    if !size_allequal(ucs)
        throw(DimensionMismatch("Unit cells provided do not have same dimensionality"))
    end
    return nothing
end

## TENSORS

function TensorKit.scalartype(
    ::Type{<:AbstractUnitCell{G,T}}
) where {G,T<:AbstractTensorMap}
    return scalartype(T)
end

# Make sure map uses efficient broadcasting
function Base.map(f, A::UnitCell{SquareSymmetric}, Bs::UnitCell{SquareSymmetric}...)
    sz = size(A)
    for B in Bs
        size(B) == sz || Base.throw_promote_shape_mismatch(sz, size(B))
    end
    return f.(A, Bs...)
end

function Broadcast.broadcasted(::UnitCellStyle{SquareSymmetric}, f, As...)
    return UnitCell{SquareSymmetric}(broadcast(f, map(_unique_only, As)...))
end
function Broadcast.broadcasted(
    ::Broadcast.AbstractArrayStyle, ::typeof(identity), A::UnitCell{SquareSymmetric}
)
    return _remove_symmetry(A)
end
function Broadcast.broadcasted(
    ::UnitCellStyle{SquareSymmetric}, ::typeof(identity), A::UnitCell{SquareSymmetric}
)
    return _remove_symmetry(A)
end
function Broadcast.broadcasted(::UnitCellUnsafeStyle{G,BStyle}, f, As...) where {G,BStyle}
    # @debug "Potentially unsafe broadcasting of $f on SquareSymmetric"
    return Broadcast.broadcasted(UnitCellStyle{G,BStyle}(), f, map(_remove_symmetry, As)...)
end

_unique_only(uc::AbstractUnitCell{SquareSymmetric}) = getdata(uc)
_unique_only(val) = val

function _remove_symmetry(uc::UnitCell{SquareSymmetric})
    mat = similar(datatype(typeof(uc)), size(uc))
    for ind in CartesianIndices(mat)
        mat[ind] = uc[ind[1], ind[2]]
    end
    return mat
end
_remove_symmetry(val) = val

Base.eachindex(uc::UnitCell{SquareSymmetric}) = eachindex(uc.data)

function Base.copyto!(dest::AbstractArray, A::Broadcast.Broadcasted{<:UnitCellStyle})
    for ind in CartesianIndices(axes(dest))
        dest[ind] = A[ind]
    end
    return dest
end
