"""
    CompositeTensor{M,T,S<:IndexSpace,N₁,N₂, ...} <: AbstractTensorMap{T,S,N₁,N₂}

Subtype of `AbstractTensorMap` for representing a M-layered tensor map, e.g. a 
tensor and it's conjugate.
"""
struct CompositeTensor{M,T,S,N₁,N₂,A<:TensorMap{T,S,N₁,N₂},F<:Tuple} <:
       AbstractTensorMap{T,S,N₁,N₂}
    layers::NTuple{M,A}
    funcs::F
end

## CONSTRUCTORS
function CompositeTensor(ct::NTuple{M,T}) where {M,T<:TensorMap}
    funcs = NTuple{M,typeof(identity)}(fill(identity, M))

    return CompositeTensor(ct, funcs)
end
function CompositeTensor(ct::NTuple{M,T}) where {M,T<:TensorKit.AdjointTensorMap}
    funcs = NTuple{M,typeof(adjoint)}(fill(adjoint, M))
    return CompositeTensor(map(parent, ct), funcs)
end
CompositeTensor(ct::NTuple{M,AbstractTensorMap}) where {M} = CompositeTensor(ct...)
function CompositeTensor(args::Vararg{<:AbstractTensorMap})
    unwrap = map(t -> t isa TensorKit.AdjointTensorMap ? parent(t) : t, args)
    conj = map(t -> t isa TensorKit.AdjointTensorMap ? adjoint : identity, args)
    return CompositeTensor(unwrap, conj)
end

"""
    CompositeTensor(f, ct::CompositeTensor)

Construct a new `CompositeTensor` by applying `mapbefore` to `ct` with the callable `f`.
"""
CompositeTensor(f, ct::CompositeTensor) = CompositeTensor(mapbefore(f, ct))

## GENERAL METHOD DEFINITIONS

Base.length(::CompositeTensor{M}) where {M} = M
Base.map(func, ct::CompositeTensor) = map((ten, af) -> func(af(ten)), ct.layers, ct.funcs)

"""
    mapbefore(f, ct::CompositeTensor{M}) -> NTuple{M}

Apply function 'f' to every tensor in `ct` *before* appying each associated composite tensor
function.
"""
mapbefore(f, ct::CompositeTensor) = map((t, af) -> af(f(t)), ct.layers, ct.funcs)

function Base.getindex(
    ct::CompositeTensor{M,T,S,N1,N2,A,F},
    ind::Union{Int128,Int16,Int32,Int64,Int8,UInt128,UInt16,UInt32,UInt64,UInt8},
) where {M,T<:Number,S<:ElementarySpace,N1,N2,A,F}
    return ct.funcs[ind](ct.layers[ind])
end
Base.firstindex(ct::CompositeTensor) = 1
Base.lastindex(ct::CompositeTensor) = length(ct)

function Base.iterate(ct::CompositeTensor, state=1)
    state > length(ct) && return nothing
    return getindex(ct, state), state + 1
end

Base.similar(ct::CompositeTensor) = CompositeTensor(mapbefore(similar, ct)...)
Base.copy!(dst::CompositeTensor, src::CompositeTensor) = (foreach(copy!, dst, src); dst)

doublelayer(t::TensorMap) = CompositeTensor((t, t))

Base.:(==)(ct1::CompositeTensor, ct2::CompositeTensor) = all(map(==, ct1, ct2))

tensortype(::Type{<:CompositeTensor{M,T,S,N₁,N₂,A}}) where {M,T,S,N₁,N₂,A} = A
TensorKit.spacetype(t::CompositeTensor) = spacetype(tensortype(t))
Base.eltype(t::CompositeTensor) = eltype(typeof(t))
Base.eltype(T::Type{<:CompositeTensor}) = eltype(tensortype(T))

TensorKit.storagetype(t::CompositeTensor) = storagetype(typeof(t))
TensorKit.storagetype(t::Type{<:CompositeTensor}) = storagetype(tensortype(t))

TensorKit.scalartype(T::Type{<:CompositeTensor}) = scalartype(tensortype(T))

## NETWORK INTERFACE

# Required
virtualspace(ct::CompositeTensor, dir::Int) = prod(mapbefore(t -> domain(t, dir), ct))
swapaxes(ct::CompositeTensor) = CompositeTensor(swapaxes, ct)
invertaxes(ct::CompositeTensor) = CompositeTensor(invertaxes, ct)
rotate(ct::CompositeTensor, i) = CompositeTensor(t -> rotate(t, i), ct)
