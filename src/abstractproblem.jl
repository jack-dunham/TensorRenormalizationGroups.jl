abstract type AbstractAlgorithm end
abstract type AbstractRuntime end

"""
$(TYPEDEF)

Concrete struct representing the state of a contraction algorithm of type `Alg` used to 
contract a network of type `Net` with runtime tensors of type `Run`. If a callback is 
provided, any returned data is stored in an instance of type `Out`. Note, avoid 
constructing this object directly, instead use the function `newproblem` to construct a 
a new instance of `InfiniteContraction`.

## Fields

$(TYPEDFIELDS)

## Constructors

    RenormalizationProblem(network::AbstractMatrix, [initial]; alg)

A new instance of `RenormalizationProblem` is constructed by passing `network` and 
(optionally) an initial runtime object, as well as the chosen `alg` as a keyword
argument. If `initial` is specified, `convertproblem` will be called to attempt to make
`initial` compatible with `alg`.

Note, `RenormalizationProblem` will *always* be constructed using a `copy` of `network`, but
*not* a `deepcopy`. That is, one can mutate the `network` struct using `setindex!` with
out mutating the constructed `RenormalizationProblem`, but mutating the tensor elements 
themselves *will* propagate through to this struct.
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
        network::Net, initial=nothing; alg::Alg
    ) where {Alg,Net<:AbstractUnitCell}
        info = ConvergenceInfo()
        runtime =
            isnothing(initial) ? initialize(network, alg) : convertproblem(Alg, initial)
        return new{Alg,Net,typeof(runtime)}(
            alg, copy(network), runtime, info, deepcopy(initial)
        )
    end
end

function RenormalizationProblem(network::AbstractMatrix, initial=nothing; kwargs...)
    uc = UnitCell(network)
    return RenormalizationProblem(uc, initial; kwargs...)
end

convertproblem(::Type, runtime) = runtime

"""
    newcontraction(algorithm, network, [initial_tensors]; kwargs...)

Initialize an instance of a `InfiniteContraction` to contract network of tensors 
`network` using algorithm `algorithm`. 

# Arguments
- `network::AbstractNetwork`: the unit cell of tensors to be contracted
- `alg::AbstractAlgorithm`: the algorithm to contract `network` with
- `initial_tensors::AbstractRuntime`: initial runtime tensors to use (optional)

# Keywords
- `store_initial::Bool = true`: if true, store a deep copy of the initial tensors
- `callback::Callback{Out} = Callback(identity, nothing)`: represents a function to be 
    executed at the end of each step

# Returns
- `ProblemState{Alg,...}`: problem state instancecorresponding to the supplied tensors 
    and parameters
"""
function newrenormalization(network; alg, kwargs...)
    initial_runtime = initialize(network, alg)
    return newrenormalization(network, initial_runtime; alg=alg, kwargs...)
end

function newrenormalization(network, initial_runtime; alg, store_initial=true, verbose=true)
    info = ConvergenceInfo()

    if store_initial
        initial_copy = deepcopy(initial_runtime)
    else
        initial_copy = nothing
    end

    prob = RenormalizationProblem(network, initial_runtime; alg=alg)

    return prob
end

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
renormalize!(problem::RenormalizationProblem; kwargs...) =
    renormalize!(identity, problem; kwargs...)
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
    reset!(problem.runtime, problem.network)
    return problem
end

"""
    recycle!(problem::ProblemState)

Reset the convergence info of `problem`.
"""
function recycle!(problem::RenormalizationProblem, network)
    continue!(problem)
    problem.info.converged = false
    problem.info.error = Inf
    problem.info.iterations = 0

    problem.network .= copy(network)

    reset!(problem.runtime, network)
    return problem
end

"""
    restart!(problem::ProblemState)

Restart the algorithm entirely, returning the tensors to their initial state.
"""
function restart!(problem::RenormalizationProblem)
    if true #isdefined(problem, :initial_tensors)
        reset!(problem)
        deepcopy!(problem.runtime, problem.initial)
    else
        println(
            "Cannot restart algorithm as initial tensors are undefined. Doing nothing..."
        )
    end
    return problem
end

"""
    forcerun!(problem::ProblemState)

Force the algorithm to continue. Equivalent to calling `continue!` followed by `run!`.
"""
function forcerun!(problem::RenormalizationProblem)
    continue!(problem)
    renormalize!(problem)
    return problem
end
