"""
$(TYPEDEF)

Stores the parameters for the variational uniform matrix product state (VUMPS) boundary algorithm. 

# Fields

$(TYPEDFIELDS)

# Constructor

    $(FUNCTIONNAME)(; bonddim, maxiter=100, tol=1e-12, verbose=true)
"""
@kwdef struct VUMPS <: AbstractBoundaryAlgorithm
    "The bond dimension of the boundary."
    bonddim::Int
    "Maximum number of iterations."
    maxiter::Int = 100
    "Convergence tolerance."
    tol::Float64 = 1e-12
    "When `true`, will print algorithm convergence progress."
    verbose::Bool = true
    function VUMPS(bonddim::Int, maxiter::Float64, tol::Float64, verbose::Bool)
        if maxiter == Inf
            maxiter = typemax(Int)
            return new(bonddim, maxiter, tol, verbose)
        else
            throw(
                ArgumentError(
                    "Unsupported `Float64` value of `maxiter`. Please supply an `Int`, or `Inf` for `maxiter = typemax(Int)`",
                ),
            )
        end
    end
    function VUMPS(bonddim::Int, maxiter::Int, tol::Float64, verbose::Bool)
        return new(bonddim, maxiter, tol, verbose)
    end
end

"""
$(TYPEDEF)

# Fields

$(TYPEDFIELDS)
"""
struct VUMPSRuntime{G,AType,CType,FType<:AbstractUnitCell,SType<:AbstractUnitCell} <:
       AbstractBoundaryRuntime
    "The boundary matrix product state (MPS)"
    mps::MPS{G,AType,CType}
    "Left and right fixed points of the transfer matrix"
    fixedpoints::FixedPoints{FType}
    "Singular values used to compute the convergence measure"
    svals::SType
end

function reset!(runtime::VUMPSRuntime, network)
    # rand!(runtime.mps)
    #
    # FL = runtime.fixedpoints.left
    # FR = runtime.fixedpoints.left
    #
    # rand!.(FL)
    # rand!.(FR)
    #
    #

    # fixedpoints!(runtime.fixedpoints, runtime.mps, network)
    return runtime
end

contraction_boundary_type(::VUMPSRuntime) = VUMPS

function Base.similar(vumps::VUMPSRuntime)
    return VUMPSRuntime(
        similar(vumps.mps), similar(vumps.fixedpoints), similar(vumps.svals)
    )
end

function TensorKit.scalartype(v::VUMPSRuntime)
    return promote_type(scalartype(v.mps), scalartype(v.fixedpoints))
end

function KrylovKit.initialize(network, alg::VUMPS)
    # D = @. getindex(domain(network), 4)

    north_bonds = virtualspace(network, 4)

    χ = dimtospace(spacetype(network), alg.bonddim)

    chi = similar(north_bonds, typeof(χ))

    chi .= Ref(χ)

    boundary_mps = MPS(randn, scalartype(network), north_bonds, chi)

    fixed_points = FixedPoints(randn, boundary_mps, network)

    svals = broadcast(getbond(boundary_mps)) do bond
        _, s, _, _ = tsvd(bond, ((1,), (2,)))
        return s
    end

    return VUMPSRuntime(boundary_mps, fixed_points, svals)
end

function step!(problem::Renormalization{<:VUMPS})
    return step!(problem.runtime, problem.network, problem.alg)
end

function step!(
    vumps::VUMPSRuntime,    #mutating
    network::AbstractNetwork,
    ::VUMPS,
)
    vumpsstep!(vumps, network; ishermitian=forcehermitian(vumps, network))

    error_per_site = boundaryerror!(vumps.svals, getbond(vumps.mps))

    @debug "Error per site:" ϵᵢ = error_per_site

    return max(error_per_site...)
end

function vumpsstep!(vumps::VUMPSRuntime, network; kwargs...)
    mps = vumps.mps
    fps = vumps.fixedpoints

    fixedpoints!(fps, mps, network; kwargs...)
    vumpsupdate!(mps, fps, network; kwargs...) # Vectorised

    return vumps
end

# Bulk of work
function vumpsupdate!(A::MPS, FP::FixedPoints, M; ishermitian=forcehermitian(A, FP, M))
    FL = FP.left
    FR = FP.right

    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    AC = getcentral(A)
    C = getbond(A)

    cummul = 1

    for x in rx

        # take mps[y], get mps[y].AC, send in mps[y].AC[x]
        # First solve for the new AC tensors.
        μ1s, ACs, _ = eigsolve(
            z -> applyhac(z, FL[x, :], FR[x, :], M[x, :]),
            AC[x, :],
            1,
            :LM;
            ishermitian=ishermitian,
            # ishermitian=false,
        )

        # @info "" ACs[1][1]
        # @info "" normalize(1 / μ1s[1] * ACs[1][1])
        # @info "" (1 / μ1s[1] * ACs[1][1])

        for y in axes(eachindex(C), 1)
            AC[x, y] = ACs[1][y]
        end
        # for y in ry
        #     a = ACs[1][y]
        #     normalize!(rmul!(a, 1 / μ1s[1]))
        #     if imag(a) ≈ a
        #         AC[x, y] = imag(a)
        #     elseif real(a) ≈ a
        #         AC[x, y] = real(a)
        #     else
        #         AC[x, y] = a
        #     end
        # end
        # now set mps[y-1, mod].AC[x] to output of above

        # A[mod(y - 1, ry)].AC[x] = ACs[1]
        μ0s, Cs, _ = eigsolve(
            z -> applyhc(z, FL[x + 1, :], FR[x, :]),
            C[x, :],
            1,
            :LM;
            ishermitian=ishermitian,
            # ishermitian=false,
        )

        # @info "" μ0s

        @debug "Individual effective Hamiltonian eigenvalues:" μ1 = μ1s[1] μ0 = μ0s[1] μ1 / μ0 = (
            μ1s[1] / μ0s[1]
        )

        for y in axes(eachindex(C), 1)
            C[x, y] = Cs[1][y]
        end
        # A[mod(y - 1, ry)].C[x] = Cs[1]
        # λ = real(μ1s[1] / μ0s[1])

        cummul *= μ1s[1] / μ0s[1]
    end

    @debug "Unit cell density" μ1 / μ0 = cummul

    #A now as updated AC, C, need to update AL, AR
    _, errL, errR = updateboth!(A)

    @debug "MPS update errors: " ϵL = findmax(errL)[1] ϵR = findmax(errR)[1]

    return A
end

# EFFECTIVE HAMILTONIANS

function applyhac(z, FL::AbstractVector, FR::AbstractVector, M::AbstractVector)
    rv = map(similar, circshift(z, -1))
    applyhac!.(rv, z, FL, FR, M)
    rv = circshift(rv, 1)
    return rv
end

function applyhc(z, FL::AbstractVector, FR::AbstractVector)
    rv = map(similar, circshift(z, -1))
    applyhc!.(rv, z, FL, FR)
    rv = circshift(rv, 1)
    return rv
end

# MPS UPDATE
# TODO: MOVE TO MPS FILE?

function updateleft!(al::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    # qac, rac = leftorth(ac)
    qac, rac = _leftorth(ac)
    qc, rc = _leftorth(c)

    normalize!(rac)
    normalize!(rc)

    mulbond!(al, qac, permute((qc'), ((), (2, 1))))
    errL = norm(rac - rc)

    return errL
end

function updateleft!(mps::MPS)
    AL, C, _, AC = mps
    errL = updateleft!.(AL, AC, C)
    return errL
end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = _rightorth(ac)
    lc, qc = _rightorth(c)

    normalize!(lac)
    normalize!(lc)

    mulbond!(ar, permute((qc'), ((), (2, 1))), qac)
    errR = norm(lac - lc)

    return errR
end

function updateright!(mps::MPS)
    _, C, AR, AC = mps
    errR = updateright!.(AR, AC, circshift(C, (1, 0)))
    return errR
end

function updateboth!(A::MPS)
    errL = updateleft!(A)
    errR = updateright!(A)
    return A, errL, errR
end

function _contract(network_top, network_bot, vumps::VUMPSRuntime, i1, i2)
    FL, FR, ACU, ARU, ACD, ARD = vumpsboundary(vumps, i1, i2)
    return _contractall(FL, FR, ACU, ARU, ACD, ARD, network_top, network_bot)
end
function _contract(network, vumps::VUMPSRuntime, i1, i2)
    FL, FR, ACU, ARU, ACD, ARD = vumpsboundary(vumps, i1, i2)
    return _contractall(FL, FR, ACU, ARU, ACD, ARD, network)
end

function vumpsboundary(vumps::VUMPSRuntime, i1, i2)
    if length(i2) > 1
        throw(
            ArgumentError(
                "
                can only contract `network` across a single row using VUMPS environment. To contract 
                across a column, compute a new environment on `permutedims(network)`.
                ",
            ),
        )
    else
        i2 = i2[begin]
    end
    _, _, AR, AC = vumps.mps

    fixed_points = vumps.fixedpoints

    l = i1[begin]
    r = i1[end]

    FL = fixed_points.left[l, i2]
    FR = fixed_points.right[r, i2]

    ACU = AC[l, i2]
    ACD = AC[l, i2 + 1]'

    ARU = tuple((AR[i, i2] for i in (l + 1):r)...)
    ARD = tuple((AR[i, i2 + 1]' for i in (l + 1):r)...)

    return FL, FR, ACU, ARU, ACD, ARD
end

### TESTING TODO: DELETE
#=
function vumps_test()
    βc = log(1 + sqrt(2)) / 2
    β = 1.0 * βc

    println("β = \t", β)

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(ComplexF64.(aM), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(aM2, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    L = Lattice{3,3,Infinite}([ℂ^2 ℂ^2; ℂ^2 ℂ^2])
    M = OnLattice(transpose(L), tM)
    # M = OnLattice(L, tM)
    # M = OnLattice(Lattice{1,1, Infinite}( fill(ℂ^2,1,1)), tM)

    alg = BoundaryAlgorithm(; alg=VUMPSTensors, bondspace=ℂ^10, verbosity=1, maxiter=200)
    vumps = inittensors(rand, M, alg)

    return state = BoundaryState(vumps, M, alg, vumps, ConvergenceInfo())

    # alg = initialise_state(VUMPS, M, BoundaryParameters(2, ℂ^4, 100, 1e-10))
    #
    # st = alg.data
    #
    # @assert isgauged(st.A[1])
    #
    # return calculate!(alg)

    # _, FL, FR = vumps!(A, M)

    # al = lefttensor(A)
    # ar = righttensor(A)
    # c = bondtensor(A)
    # ac = centraltensor(A)

    # z2 = expval(tM, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
    # magn = expval(tM2, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))

    # return magn ./ z2
end

# function testflsolve(FL, P, A)
#     @tensoropt FLo[k4 r4; r3 k3] :=
#         FL[k2 r2; r1 k1] *
#         A[k1 D1; k3 D3] *
#         P[r1 D2 d1; r3 D1 d2] *
#         (P')[r4 D3 d2; r2 D4 d1] *
#         (A')[k4 D4; k2 D2]
#     return Flo
# end

=#
