const AbstractNetwork{G,ElType<:AbstractTensorMap,A} = AbstractUnitCell{G,ElType,A}
const AbstractSingleLayerNetwork{G,ElType<:TensorMap,A} = AbstractUnitCell{G,ElType,A}
const AbstractDoubleLayerNetwork{G,ElType<:CompositeTensor{2},A} = AbstractUnitCell{
    G,ElType,A
}

TensorKit.spacetype(uc::AbstractUnitCell) = spacetype(typeof(uc))
TensorKit.spacetype(::Type{<:AbstractNetwork{G,T}}) where {G,T} = spacetype(T)
## Implement functions for contractable tensors etc
## ContractableTensors need a notion of an east, south, west, north bonds

virtualspace(network::AbstractUnitCell, dir) = virtualspace.(network, dir)

swapaxes(network::AbstractNetwork) = swapaxes.(network)
invertaxes(network::AbstractNetwork) = invertaxes.(network)

## INTERFACE

"""
    $(FUNCTIONNAME)(tensor, [dir::Integer])

Return an `NTuple{4,<:VectorSpace}` containing the east, south, west, and north vector 
spaces associated with the respective bonds, in that order. If `dir` is provided, then
return the corresponding `VectorSpace` in that tuple.
!!! note
    For custom data types, this function must be specified for use in contraction algorithms.
"""
virtualspace(t::AbsTen{0,4}, dir) = domain(t, dir)
virtualspace(t) = map(i -> virtualspace(t, i), (1, 2, 3, 4))

"""
    $(FUNCTIONNAME)(tensor)

Swap the horizontal and the vertical virtual bonds of an object.

!!! note 
    For custom data types, this function must be specified for use in contraction algorithms.
"""
swapaxes(t::TenAbs{4}) = permutedom(t, (2, 1, 4, 3))

"""
    $(FUNCTIONNAME)(tensor)

Invert the horizontal and the vertical bonds of an object `t`, that is, south ↔ north and 
east ↔ west. 
!!! note
    For custom data types, this function must be specified for use in contraction algorithms.
"""
invertaxes(t::TenAbs{4}) = permutedom(t, (3, 4, 1, 2))

# Deprec
ensure_contractable(x) = x
