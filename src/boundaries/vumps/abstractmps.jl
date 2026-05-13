# DONE
abstract type AbstractMPS end

"""
    struct $(FUNCTIONNAME){G<:AbstractUnitCellGeometry, AType<:AbstractTensorMap{N, 2}, CType<:AbstractTensorMap{0, 2}}

Infinite boundary matrix product state (MPS) in mixed canonical form. MPS tensors have `N`
"physical" indices.
Has fields `AL`, `C`, `AR` and derived field `AC` such that
```math
A_L(x,y) C(x,y) \\approx C(x - 1, y) A_R(x, y) \\approx AC(x, y) \\quad \\forall x,y \\in \\Lambda
```
where ``\\Lambda`` defines the unit cell of geometry `G`.

# Examples

```@repl
using TensorKit
physbonds = UnitCell(fill(ComplexSpace(2), 2, 1))
virtbonds = UnitCell(fill(ComplexSpace(3), 2, 1))
mps = MPS(rand, ComplexF64, physbonds, virtbonds);
AL, C, AR, AC = mps;
C
```

"""
struct MPS{G,AType<:TenAbs{2},CType<:AbsTen{0,2}} <: AbstractMPS
    AL::UnitCell{G,AType,Matrix{AType}}
    C::UnitCell{G,CType,Matrix{CType}}
    AR::UnitCell{G,AType,Matrix{AType}}
    AC::UnitCell{G,AType,Matrix{AType}}
    function MPS(
        AL::UA, C::UC, AR::UA, AC::UA
    ) where {G,AType,UA<:AbstractUnitCell{G,AType},CType,UC<:AbstractUnitCell{G,CType}}
        check_size_allequal(AL, C, AR, AC)
        mps = new{G,AType,CType}(AL, C, AR, AC)
        validate(mps)
        return mps
    end
    # function MPS(
    #     AL::AType, C::CType, AR::AType, AC::AType
    # ) where {AType<:SubUnitCell,CType<:SubUnitCell}
    #     mps = new{AType,CType}(AL, C, AR, AC)
    #     return mps
    # end
end

# FIELD GETTERS

getleft(mps::MPS) = mps.AL
getbond(mps::MPS) = mps.C
getright(mps::MPS) = mps.AR
getcentral(mps::MPS) = mps.AC

unpack(mps::AbstractMPS) = (getleft(mps), getbond(mps), getright(mps), getcentral(mps))

function TensorKit.scalartype(mps::MPS{G,AType,CType}) where {G,AType,CType}
    return promote_type(scalartype(AType), scalartype(AType))
end

# BASE

Base.size(mps::AbstractMPS) = size(getcentral(mps))
Base.length(mps::AbstractMPS) = 4
Base.similar(mps::M) where {M<:MPS} = MPS(similar.(mps)...)::M

function TensorKit.rand!(mps::MPS)
    AL, C, AR, AC = mps
    rand!.(AC)
    mixedgauge!(AL, C, AR, AC)
    return mps
end

function Base.iterate(mps::AbstractMPS, state=1)
    if state > 4
        return nothing
    else
        return unpack(mps)[state], state + 1
    end
end
Base.transpose(M::MPS) = map(transpose, unpack(M))

# GAUGING

function isgauged(mps::AbstractMPS)
    ch_isassigned = arr -> all(x -> x, isassigned.(Ref(arr), CartesianIndices(arr)))

    if all(x -> ch_isassigned(x), mps)
        AL, C, AR, AC = mps
        for y in axes(AL, 2)
            for x in axes(AL, 1)
                if !(mulbond(AL[x, y], C[x, y]) ≈ mulbond(C[x - 1, y], AR[x, y]) ≈ AC[x, y])
                    return 0
                end
            end
        end
    else
        return -1
    end

    return 1
end

function validate(mps::AbstractMPS)
    isg = isgauged(mps)
    if isg == 0
        @warn "MPS is not in mixed canonical form."
    elseif isg == -1
        @debug "Undefined MPS. Skipping gauge check..."
    elseif isg == 1
        # @debug "MPS is in mixed canonical form."
    end
    return mps
end

# CONSTRUCTORS

function MPS(
    f, T::Type{<:Number}, physbonds::AbstractUnitCell, rightbonds::AbstractUnitCell
)
    leftbonds = circshift(rightbonds, (1, 0))
    data_lat = @. f(T, physbonds, rightbonds * adjoint(leftbonds))
    return MPS(data_lat)
end

function MPS(A::AbstractUnitCell)
    bonds = @. getindex(domain(A), 1) # Canonical bond (right-hand) bond of mps tensor

    T = scalartype(A)

    AL = similar.(A)
    C = @. rand(T, one(bonds), bonds * adjoint(bonds)) # R * L
    AR = similar.(A)

    mixedgauge!(AL, C, AR, A)
    return MPS(AL, C, AR)
end

MPS(AL, C, AR) = MPS(AL, C, AR, centraltensor(AL, C))

# TENSOR OPERATIONS

function _similar_ac(
    ac1::AbstractTensorMap{T,S,N}, ac2::AbstractTensorMap{T,S,M}
) where {T,S,N,M}
    return _similar_ac(Val(N), ac1, ac2)
end
_similar_ac(::Val{0}, ac1, ac2) = similar(ac2)
_similar_ac(::Val{N}, ac1, ac2) where {N} = similar(ac1)

centraltensor(A1, A2) = centraltensor!(_similar_ac.(A1, A2), A1, A2)
function centraltensor!(AC, AL::AbUnCe{<:TenAbs{2}}, C)
    return mulbond!.(AC, AL, C)
end
function centraltensor!(AC, C::AbUnCe{<:AbsTen{0,2}}, AR)
    return mulbond!.(AC, circshift(C, (1, 0)), AR)
end

function mulbond(A1::AbstractTensorMap{T,S}, A2::AbstractTensorMap{T,S}) where {T,S}
    return mulbond!(_similar_ac(A1, A2), A1, A2)
end
# Right
function mulbond!(
    CA::AbsTen{1,2,S}, C::AbsTen{0,2,S}, A::AbsTen{1,2,S}
) where {S<:IndexSpace}
    return @tensoropt CA[p1; xr xl] = C[x_in xl] * A[p1; xr x_in]
end
function mulbond!(
    CA::AbsTen{2,2,S}, C::AbsTen{0,2,S}, A::AbsTen{2,2,S}
) where {S<:IndexSpace}
    return @tensoropt CA[p1 p2; xr xl] = C[x_in xl] * A[p1 p2; xr x_in]
end
# Left
function mulbond!(
    AC::AbsTen{1,2,S}, A::AbsTen{1,2,S}, C::AbsTen{0,2,S}
) where {S<:IndexSpace}
    return @tensoropt AC[p1; xr xl] = A[p1; x_in xl] * C[xr x_in]
end
function mulbond!(
    AC::AbsTen{2,2,S}, A::AbsTen{2,2,S}, C::AbsTen{0,2,S}
) where {S<:IndexSpace}
    return @tensoropt AC[p1 p2; xr xl] = A[p1 p2; x_in xl] * C[xr x_in]
end
