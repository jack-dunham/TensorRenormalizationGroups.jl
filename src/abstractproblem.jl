abstract type AbstractAlgorithm end
abstract type AbstractRuntime end

"""
$(TYPEDEF)

Concrete struct representing the state of a contraction algorithm of type `Alg` used to 
contract a network of type `Net` with runtime tensors of type `Run`. 

## Fields

$(TYPEDFIELDS)

## Constructors

    RenormalizationProblem(network::AbstractMatrix, alg::AbstractAlgorithm, [initial::AbstractRuntime])

A new instance of `RenormalizationProblem` is constructed by passing `network` and 
(optionally) an initial runtime object, as well as the chosen `alg`. 
If `initial` is specified, `convertproblem` will be called to attempt to make
`initial` compatible with `alg`.

Note, `RenormalizationProblem` will *always* be constructed using a `copy` of `network`, but
*not* a `deepcopy`. That is, one can mutate the `network` struct using `setindex!` with
out mutating the constructed `RenormalizationProblem`, but mutating the tensor elements 
themselves *will* propagate through to this struct.

    RenormalizationProblem(network::AbstractMatrix, problem::RenormalizationProblem)

Constuct a new instance of `RenormalizationProblem` using the algorithm and runtime from
existing `problem`.
"""
struct RenormalizationProblem{
    Alg<:AbstractAlgorithm,Net<:AbstractUnitCell,Run<:AbstractRuntime
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
    function RenormalizationProblem(
        network::Net, alg::Alg, initial
    ) where {Alg<:AbstractAlgorithm,Net<:AbstractUnitCell}
        info = ConvergenceInfo()
        if isnothing(initial)
            @info "he"
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
function RenormalizationProblem(
    network::AbstractMatrix, alg::AbstractAlgorithm, initial=nothing
)
    return RenormalizationProblem(UnitCell(network), alg, initial)
end
function RenormalizationProblem(network::AbstractMatrix, problem::RenormalizationProblem)
    return RenormalizationProblem(network, problem.alg, problem.runtime)
end

convertproblem(::Type, runtime) = runtime

function _run!(callback, problem::RenormalizationProblem)
    info = problem.info
    alg = problem.alg

    @info "Running algorithm:" algorithm = alg

    while info.iterations < alg.maxiter && info.error ≥ alg.tol
        info.error = step!(problem)

        info.iterations += 1

        if alg.verbose
            @info "Convergence ≈ $(info.error) after $(info.iterations) iterations."
        end

        callback(problem)
    end
    info.error > alg.tol ? info.converged = false : info.converged = true

    info.finished = true

    @info "Convergence: $(info.error)"
    if info.converged
        @info "Algorithm convergenced to within tolerance $(alg.tol) after $(info.iterations) iterations"
    else
        @warn "Algorithm did not convergence to within $(alg.tol) after $(info.iterations) iterations"
    end

    return info.finished
end

"""
    runcontraction(problem::ProblemState)

Equivalent to `runcontraction!(deepcopy(problem))`.
"""
renormalize(problem; kwargs...) = renormalize!(deepcopy(problem); kwargs...)

"""
    renormalize!([callback=identity,] problem::TensorRenormalizationProblem; kwargs...)

Perform the renormalization defined in `problem` executing `callback(problem)` after
each step and mutating `problem` in place. If `verbose = false`, top-level information
about algorithm progress will be surpressed.
"""
function renormalize!(problem::RenormalizationProblem; kwargs...)
    return renormalize!(identity, problem; kwargs...)
end
function renormalize!(callback, problem::RenormalizationProblem; kwargs...)
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
    continue!(problem::ProblemState)

Allow `problem` to continue past the termination criteria.
"""
function continue!(problem::RenormalizationProblem)
    problem.info.finished = false
    return problem
end

"""
    reset!(problem::ProblemState)

Reset the convergence info of `problem`.
"""
function reset!(problem::RenormalizationProblem)
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
    recycle!(problem::ProblemState)

Call `reset!` on `problem`, and set the elements `problem.network` to that of `copy(network)`
"""
function recycle!(problem::RenormalizationProblem, network)
    problem.network .= copy(network)
    reset!(problem)
    return problem
end

"""
    restart!(problem::ProblemState)

Restart the algorithm entirely, returning the tensors to their initial state.
"""
function restart!(problem::RenormalizationProblem)
    reset!(problem)
    deepcopy!(problem.runtime, problem.initial)
    return problem
end
