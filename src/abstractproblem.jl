abstract type AbstractRenormalizationAlgorithm end
abstract type AbstractRenormalizationRuntime end

"""
$(TYPEDEF)

Concrete struct representing the state of a contraction algorithm of type `Alg` used to 
contract a network of type `Net` with runtime tensors of type `Run`. 

# Fields

$(TYPEDFIELDS)

# Constructors

    Renormalization(network::AbstractMatrix, alg::AbstractRenormalizationAlgorithm, [initial::AbstractRenormalizationRuntime])

A new instance of `Renormalization` is constructed by passing `network` and 
(optionally) an initial runtime object, as well as the chosen `alg`. 
If `initial` is specified, `convertproblem` will be called to attempt to make
`initial` compatible with `alg`.

Note, `Renormalization` will *always* be constructed using a `copy` of `network`, but
*not* a `deepcopy`. That is, one can mutate the `network` struct using `setindex!` with
out mutating the constructed `Renormalization`, but mutating the tensor elements 
themselves *will* propagate through to this struct.

    Renormalization(network::AbstractMatrix, problem::Renormalization)

Constuct a new instance of `Renormalization` using the algorithm and runtime from
existing `problem`.
"""
struct Renormalization{
    Alg<:AbstractRenormalizationAlgorithm,
    Net<:AbstractUnitCell,
    Run<:AbstractRenormalizationRuntime,
}
    "The renormalization algorithm to contract `network` with."
    alg::Alg
    "An `AbstractUnitCell` representing the network of tensors to be contracted."
    network::Net
    "The runtime object corresponding to `alg`."
    runtime::Run
    "Convegence info for this renormalization instance."
    info::ConvergenceInfo
    "A `deepcopy` of `runtime` called at construction."
    initial::Run
    function Renormalization(
        network::Net, alg::Alg, initial
    ) where {Alg<:AbstractRenormalizationAlgorithm,Net<:AbstractUnitCell}
        info = ConvergenceInfo()
        if isnothing(initial)
            runtime = initialize(network, alg)
        else
            runtime = convertproblem(Alg, initial)
        end
        @info typeof(runtime)
        return new{Alg,Net,typeof(runtime)}(
            alg, copy(network), runtime, info, deepcopy(runtime)
        )
    end
end
function Renormalization(
    network::AbstractMatrix, alg::AbstractRenormalizationAlgorithm, initial=nothing
)
    return Renormalization(UnitCell(network), alg, initial)
end
function Renormalization(network::AbstractMatrix, problem::Renormalization)
    return Renormalization(network, problem.alg, problem.runtime)
end

function TensorKit.scalartype(::Type{<:Renormalization{Alg,Net}}) where {Alg,Net}
    return scalartype(Net)
end

convertproblem(::Type, runtime) = runtime

function _run!(callback, problem::Renormalization)
    info = problem.info
    alg = problem.alg
    TAlg = typeof(alg)

    @info "Running algorithm:" algorithm = alg

    while info.iterations < alg.maxiter && info.error ≥ alg.tol
        info.error = step!(problem)

        info.iterations += 1

        if alg.verbose
            @info "$TAlg convergence ≈ $(info.error) after $(info.iterations) iterations."
        end

        callback(problem)
    end
    info.error > alg.tol ? info.converged = false : info.converged = true

    info.finished = true

    @info "Convergence: $(info.error)"
    if info.converged
        @info "$TAlg convergenced to within tolerance $(alg.tol) after $(info.iterations) iterations"
    else
        @warn "$TAlg did not convergence to within $(alg.tol) after $(info.iterations) iterations"
    end

    return info.finished
end

"""
    $(FUNCTIONNAME)([callback=identity,] problem::Renormalization; kwargs...)

Perform the renormalization defined in `problem` executing `callback(problem)` after
each step and mutating `problem` in place. If `verbose = false`, top-level information
about algorithm progress will be surpressed.
"""
function renormalize!(problem::Renormalization; kwargs...)
    return renormalize!(identity, problem; kwargs...)
end
function renormalize!(callback, problem::Renormalization; kwargs...)
    if problem.info.finished == true
        println(
            "Problem has reached termination according to parameters set. Use `forcerun!`, 
                or `continue!` followed by `runcontraction!` to ignore this and continue anyway.",
        )
    else
        _run!(callback, problem; kwargs...)
    end
end

"""
    $(TYPEDSIGNATURES)

Allow `problem` to continue past the termination criteria.
"""
function continue!(problem::Renormalization)
    problem.info.finished = false
    return problem
end

"""
$(TYPEDSIGNATURES)

Reset the convergence info of `problem`.
"""
function reset!(problem::Renormalization)
    continue!(problem)
    problem.info.converged = false
    problem.info.error = Inf
    problem.info.iterations = 0
    # Some runtime objects store fields derived from `problem.network`. Need to update
    # these.
    reset!(problem.runtime, problem.network)
    return problem
end

"""
$(TYPEDSIGNATURES)

Call `reset!` on `problem`, and set the elements `problem.network` to that of `copy(network)`
"""
function recycle!(problem::Renormalization, network)
    problem.network .= copy(network)
    reset!(problem)
    return problem
end

"""
$(TYPEDSIGNATURES)

Restart the algorithm entirely, returning the tensors to their initial state.
"""
function restart!(problem::Renormalization)
    reset!(problem)
    deepcopy!(problem.runtime, problem.initial)
    return problem
end
