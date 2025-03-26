abstract type AbstractUnitCellGeometry end
abstract type AbstractUnitCell{G<:AbstractUnitCellGeometry,ElType,A} <:
              AbstractMatrix{ElType} end

const AbUnCe{T,G,A} = AbstractUnitCell{G,T,A}

struct Square <: AbstractUnitCellGeometry end
struct SquareSymmetric <: AbstractUnitCellGeometry end

struct UnitCell{G<:AbstractUnitCellGeometry,ElType,A} <: AbstractUnitCell{G,ElType,A}
    data::CircularArray{ElType,2,A}
    UnitCell{G,T,A}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
    UnitCell{G}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
end

@inline getdata(uc::UnitCell{G,T,A}) where {G,T,A} = uc.data::CircularArray{T,2,A}

@inline datatype(U::Type{<:AbstractUnitCell{G,ElType,A}}) where {G,ElType,A} = A
@inline datatype(U::Type) = U

## ABSTRACT ARRAY

@inline Base.size(uc::AbstractUnitCell) = size(getdata(uc))
function Base.size(uc::AbstractUnitCell{SquareSymmetric})
    nx = size(getdata(uc), 1)
    return (nx, nx)
end
Base.getindex(uc::AbstractUnitCell{G}, i...) where {G} = _getindex(G, uc, i...)
_getindex(::Type{<:AbstractUnitCellGeometry}, uc, i...) = getindex(getdata(uc), i...)
@inline Base.setindex!(uc::AbstractUnitCell, v, i...) = setindex!(getdata(uc), v, i...)

function _getindex(G::Type{SquareSymmetric}, uc::AbstractUnitCell, inds::Tuple)
    return _getindex(G, uc, inds[1], inds[2])
end
function _getindex(::Type{SquareSymmetric}, uc::AbstractUnitCell, i::Int)
    nx = size(uc, 1)
    q = div(i - 1, nx, RoundDown)
    r = mod(i, 1:3)
    return getindex(circshift(getdata(uc), -q), r)
end
function _getindex(::Type{SquareSymmetric}, uc::AbstractUnitCell, i1, i2::Int)
    return getindex(circshift(getdata(uc), -(i2 - 1)), i1)
end
function _getindex(G::Type{SquareSymmetric}, uc::AbstractUnitCell, i1, i2)
    return _getindex(G, uc, i2, i1)
end
## 

## SIMILAR

@inline function Base.similar(uc::UnitCell{G,T}) where {G,T}
    return UnitCell{G}(similar(getdata(uc)))
end
@inline function Base.similar(uc::UnitCell{G,T}, ::Type{S}) where {G,S,T}
    return UnitCell{G}(similar(getdata(uc), S))
end
@inline function Base.similar(uc::UnitCell{G,T}, dims::Dims) where {T,G}
    return UnitCell{G}(similar(getdata(uc), dims))
end
@inline function Base.similar(uc::UnitCell{G,T}, ::Type{S}, dims::Dims) where {T,S,G}
    return UnitCell{G}(similar(getdata(uc), S, dims))
end

##

## BROADCASTING

# Pass in the `BroadcastStyle` such that we can get compute the winning broadcast style
# of the underlying abstract matrix.

abstract type AbstractUnitCellStyle{G,A<:Base.BroadcastStyle} <:
              Broadcast.AbstractArrayStyle{2} end

struct UnitCellStyle{G,A} <: AbstractUnitCellStyle{G,A} end

(T::Type{<:AbstractUnitCellStyle})(::Val{2}) = T()
(T::Type{<:AbstractUnitCellStyle})(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

@inline function Broadcast.BroadcastStyle(::Type{UnitCell{G,ElType,A}}) where {G,ElType,A}
    return UnitCellStyle{G,typeof(Broadcast.BroadcastStyle(A))}()
end
@inline function Broadcast.BroadcastStyle(
    ::Type{<:AbstractUnitCell{G,ElType,A}}
) where {G,ElType,A}
    return UnitCellStyle{G,typeof(Broadcast.BroadcastStyle(A))}()
end

function Broadcast.BroadcastStyle(
    ::UnitCellStyle{G,A}, ::Broadcast.ArrayStyle{<:CircularArray{ElType,2,B}}
) where {G,ElType,A,B}
    AB = Broadcast.BroadcastStyle(A(), Broadcast.BroadcastStyle(B))
    return UnitCellStyle{G,typeof(AB)}()
end

function Broadcast.BroadcastStyle(
    ::UnitCellStyle{SquareSymmetric}, G::Broadcast.DefaultArrayStyle{2}
)
    return G
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

@inline function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return UnitCell{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end

## CONSTRUCTORS

tocircular(data::AbstractMatrix) = CircularArray(data)
tocircular(data::CircularArray) = data
tocircular(data::AbstractUnitCell) = getdata(data)

# Make circular
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
