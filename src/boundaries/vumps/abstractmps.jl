# DONE
abstract type AbstractMPS end

@doc raw"""
    struct MPS
"""
struct MPS{AType<:AbUnCe{<:TenAbs{2}},CType<:AbUnCe{<:AbsTen{0,2}}} <: AbstractMPS
    AL::AType
    C::CType
    AR::AType
    AC::AType
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType
    ) where {AType<:AbstractUnitCell,CType<:AbstractUnitCell}
        check_size_allequal(AL, C, AR, AC)
        mps = new{AType,CType}(AL, C, AR, AC)
        validate(mps)
        return mps
    end
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType
    ) where {AType<:SubUnitCell,CType<:SubUnitCell}
        mps = new{AType,CType}(AL, C, AR, AC)
        return mps
    end
end

# FIELD GETTERS

getleft(mps::MPS) = mps.AL
getbond(mps::MPS) = mps.C
getright(mps::MPS) = mps.AR
getcentral(mps::MPS) = mps.AC

unpack(mps::AbstractMPS) = (getleft(mps), getbond(mps), getright(mps), getcentral(mps))

TensorKit.scalartype(mps::AbstractMPS) = promote_type(map(scalartype, unpack(mps))...)

# BASE

Base.size(mps::AbstractMPS) = size(getcentral(mps))
Base.length(mps::AbstractMPS) = size(mps)[2]
Base.similar(mps::M) where {M<:MPS} = MPS(similar.(unpack(mps))...)::M

function TensorKit.rand!(mps::MPS)
    AL, C, AR, AC = unpack(mps)
    rand!.(AC)
    mixedgauge!(AL, C, AR, AC)
    return mps
end

function Base.getindex(mps::AbstractMPS, y::Int)
    AL, C, AR, AC = unpack(mps)
    @views begin
        subAL = AL[:, y]
        subC = C[:, y]
        subAR = AR[:, y]
        subAC = AC[:, y]
    end
    return MPS(subAL, subC, subAR, subAC)
end

function Base.iterate(mps::AbstractMPS, state=1)
    if state > length(mps)
        return nothing
    else
        return mps[state], state + 1
    end
end

function Base.transpose(M::MPS)
    return transpose.(unpack(M))
end

# GAUGING

function isgauged(mps::AbstractMPS)
    ch_isassigned = arr -> all(x -> x, isassigned.(Ref(arr), CartesianIndices(arr)))

    for single_mps in mps
        unp = unpack(single_mps)
        if all(x -> ch_isassigned(x), unp)
            AL, C, AR, AC = unp
            for x in axes(AL, 1)
                if !(mulbond(AL[x], C[x]) ≈ mulbond(C[x - 1], AR[x]) ≈ AC[x])
                    return 0
                end
            end
        else
            return -1
        end
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

# function MPS(f, T, lattice, D, χ; kwargs...)
#     return MPS(f, T, _fill_all_maybe(lattice, D, χ)...; kwargs...)
# end

function MPS(f, T, D::AbstractUnitCell, right_bonds::AbstractUnitCell)
    left_bonds = circshift(right_bonds, (1, 0))
    data_lat = @. f(T, D, right_bonds * adjoint(left_bonds))
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
