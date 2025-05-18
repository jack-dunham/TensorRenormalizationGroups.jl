module InfiniteTensorContractions
const ITC = InfiniteTensorContractions

using CircularArrays
using KrylovKit
using LinearAlgebra
using TensorKit
using DocStringExtensions

export ITC

export AbstractUnitCellGeometry
export AbstractUnitCell

# RenormalizationProblem
export RenormalizationProblem

# ABSTRACT RUNTIMES
export AbstractRuntime
export AbstractBoundaryRuntime, AbstractGrainingRuntime

# ABSTRACT ALGORITHMS
export AbstractAlgorithm
export AbstractBoundaryAlgorithm, AbstractGrainingAlgorithm

export Square, SquareSymmetric
export UnitCell
export CompositeTensor, virtualspace, swapaxes, invertaxes

# ALGORITHMS
export VUMPS, CTMRG, TRG

# VUMPS
export VUMPSRuntime, MPS, FixedPoints, TransferMatrix

# CORNERS
export CornerMethodTensors, CornerMethodRuntime, Corners, Edges
export corners, edges

export getboundary

# INTERFACE
export newrenormalization, initialize, renormalize!, renormalize, contract

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

end # module
