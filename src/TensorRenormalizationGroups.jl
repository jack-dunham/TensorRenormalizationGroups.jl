"""
*A Julia package implementing various two-dimensional tensor renormalization group-like
algorithms under a single interface. Designed to be application agnostic. 
Based on [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).*

!!! note
    This documentation is a work in progress.
"""
module TensorRenormalizationGroups
const TRGroups = TensorRenormalizationGroups

using CircularArrays
using KrylovKit
using LinearAlgebra
using TensorKit
using Printf
using DocStringExtensions

export TRGroups

export AbstractUnitCellGeometry
export AbstractUnitCell

# Renormalization
export Renormalization

# ABSTRACT RUNTIMES
export AbstractRenormalizationRuntime
export AbstractBoundaryRuntime, AbstractGrainingRuntime

# ABSTRACT ALGORITHMS
export AbstractRenormalizationAlgorithm
export AbstractBoundaryAlgorithm, AbstractGrainingAlgorithm

export Square, SquareSymmetric
export UnitCell
export tensortype
export CompositeTensor, virtualspace, swapaxes, invertaxes

# ALGORITHMS
export VUMPS, CTMRG, TRG

# VUMPS
export VUMPSRuntime, MPS, FixedPoints, TransferMatrix

# CORNERS
export AbstractCornerMethod
export CornerMethodTensors, CornerMethodRuntime, Corners, Edges
export corners, edges

# GRAINING
export TRGRuntime

export getboundary

# INTERFACE
export renormalize!, contract
public continue!, reset!, recycle!, restart!

# No deps
include("convergenceinfo.jl")
include("utils.jl")
include("callback.jl")

include("compositetensor.jl")
include("abstractunitcell.jl")
include("abstractproblem.jl")
include("networks.jl")

include("contract.jl")

include("boundaries/abstractboundary.jl")

# VUMPS
include("boundaries/vumps/abstractmps.jl")
include("boundaries/vumps/mpsgauge.jl")

include("boundaries/vumps/tensormacros.jl")
include("boundaries/vumps/transfermatrix.jl")
include("boundaries/vumps/fixedpoints.jl")
include("boundaries/vumps/vumps.jl")

# CTMRG
include("boundaries/corner/tensormacros.jl")
include("boundaries/corner/cornermethod.jl")
include("boundaries/corner/ctmrg.jl")
include("boundaries/corner/init.jl")

# FPCM
include("boundaries/corner/fpcm.jl")
include("boundaries/corner/biorth.jl")

include("boundaries/corner/hybrid.jl")

# GRAINING
include("graining/abstractgraining.jl")
include("graining/trg.jl")

function __init__()
    ENV["ITC_ALWAYS_FORCE"] = false
    return nothing
end

function testpeps(x)
    β = x * log(1 + sqrt(2)) / 2

    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
    X = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2)

    op = exp(β / 2 * Z ⊗ Z)

    U, s, V = tsvd(op, (1, 3), (2, 4); trunc=truncbelow(eps()))

    E = S = U * sqrt(s)
    W = N = sqrt(s) * V

    plus = Tensor(1 / sqrt(2) * [1, -1], ℂ^2)

    @tensoropt state[out; e s w n] :=
        E[out x4; e] * S[x4 x3; s] * W[w; x3 x2] * N[n; x2 x1] * plus[x1]

    hs = TensorMap{Float64}(
        undef, one(ComplexSpace), ℂ^2 * (ℂ^2)' * ℂ^2 * (ℂ^2)' * (ℂ^2)' * ℂ^2 * (ℂ^2)' * ℂ^2
    )
    hs = @tensor hs[e1 e2 s1 s2 w1 w2 n1 n2] =
        one(Z)[x2; x1] * state[x1; e1 s1 w1 n1] * (state')[e2 s2 w2 n2; x2]
    hsm = TensorMap(ComplexF64.(hs.data), one(ComplexSpace), ℂ^4 * ℂ^4 * (ℂ^4)' * (ℂ^4)')
    net = UnitCell(fill(hsm, 1, 1))

    hsZ = TensorMap{Float64}(
        undef, one(ComplexSpace), ℂ^2 * (ℂ^2)' * ℂ^2 * (ℂ^2)' * (ℂ^2)' * ℂ^2 * (ℂ^2)' * ℂ^2
    )
    hsZ = @tensor hsZ[e1 e2 s1 s2 w1 w2 n1 n2] =
        Z[x2; x1] * state[x1; e1 s1 w1 n1] * (state')[e2 s2 w2 n2; x2]
    hsmZ = TensorMap(hsZ.data, one(ComplexSpace), ℂ^4 * ℂ^4 * (ℂ^4)' * (ℂ^4)')
    netZ = UnitCell(fill(hsmZ, 1, 1))

    @info scalartype.(net)
    rt = Renormalization(net, VUMPS(; maxiter=100, bonddim=20))

    renormalize!(rt)
    norm = contract(net, rt.runtime)[1, 1]

    magn_ss = contract(netZ, rt.runtime)[1, 1] / norm

    @info "" norm magn_ss

    rv = []

    for l in 2:20
        bulk = fill(hsm, l - 2)

        magn = contract([hsmZ, bulk..., hsmZ], rt.runtime, 1:l, 1:1)
        magn_norm = contract([hsm, bulk..., hsm], rt.runtime, 1:l, 1:1)

        push!(rv, magn / magn_norm - magn_ss^2)
    end

    return rv
end

function testpeps2(x)
    β = x * log(1 + sqrt(2)) / 2

    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
    X = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2)

    op = exp(β / 2 * Z ⊗ Z)

    U, s, V = tsvd(op, (1, 3), (2, 4); trunc=truncbelow(eps()))

    E = S = U * sqrt(s)
    W = N = sqrt(s) * V

    plus = Tensor(1 / sqrt(2) * [1, -1], ℂ^2)

    @tensoropt state[out; e s w n] :=
        E[out x4; e] * S[x4 x3; s] * W[w; x3 x2] * N[n; x2 x1] * plus[x1]

    size = 2

    net = UnitCell(fill(CompositeTensor(state, state'), size, size))
    netZ = UnitCell(fill(CompositeTensor(state, (Z * state)'), size, size))

    # hsZ = TensorMap{Float64}(
    #     undef, one(ComplexSpace), ℂ^2 * (ℂ^2)' * ℂ^2 * (ℂ^2)' * (ℂ^2)' * ℂ^2 * (ℂ^2)' * ℂ^2
    # )
    # hsZ = @tensor hsZ[e1 e2 s1 s2 w1 w2 n1 n2] =
    #     Z[x2; x1] * state[x1; e1 s1 w1 n1] * (state')[e2 s2 w2 n2; x2]
    # hsmZ = TensorMap(hsZ.data, one(ComplexSpace), ℂ^4 * ℂ^4 * (ℂ^4)' * (ℂ^4)')
    # netZ = UnitCell(fill(hsmZ, 1, 1))

    @info scalartype.(net)
    rt = Renormalization(net, CTMRG(; maxiter=100, bonddim=40))

    renormalize!(rt)
    norm = contract(net, rt.runtime)[1, 1]

    @info norm

    magn_ss = contract(netZ, rt.runtime) ./ norm

    @info "" norm magn_ss

    magn_ss = magn_ss[1, 1]

    rv = []

    for l in 2:20
        bulk = fill(state, l - 2)

        ops = [Z * state, bulk..., Z * state]
        bare_ops = [state, bulk..., state]

        magn_in = map(CompositeTensor, ops, adjoint.(bare_ops))
        magn_norm_in = map(CompositeTensor, bare_ops, adjoint.(bare_ops))

        magn = contract(magn_in, rt.runtime, 1:l, 1)
        magn_norm = contract(magn_norm_in, rt.runtime, 1:l, 1)

        push!(rv, magn / magn_norm - magn_ss^2)
    end

    return rv
end
end # module
