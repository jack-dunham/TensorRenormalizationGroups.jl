struct TRGRuntime{AType,SType} <: AbstractGrainingRuntime
    tensors::AType
    svals::SType
    cumsum::Matrix{Float64}
end

@kwdef struct TRG{T} <: AbstractAlgorithm
    trunc::T
    maxiter::Int = 20
    tol::Float64 = 1e-12
    verbose::Bool = true
end

function KrylovKit.initialize(network, ::TRG)
    nx, ny = size(network)

    Nx = lcm(2, nx)
    Ny = lcm(2, ny)

    inds = Iterators.product(1:Nx, 1:Ny)

    svals = broadcast(inds) do (x, y)
        t = network[x, y]
        s = spacetype(t)
        sval = similar(t, oneunit(s), oneunit(s))
        return sval
    end

    tensors = broadcast(network) do t
        rv = copy(t)
        return rv
    end

    cumsum = zeros(size(network))

    return TRGRuntime(tensors, svals, cumsum)
end

function trg_unitaries(svds, ind)
    x = ind[1]
    y = ind[2]

    # For every y we move back a x
    xp = x - (y - 1)

    # For every x we move forward a y
    yp = y + (x - 1)

    tl = svds[xp, yp].v # 3 v1
    tr = svds[xp + 1, yp].v # 4 v2
    bl = svds[xp + 1, xp].u # 2 u2
    br = svds[xp + 1, yp + 1].u # 1 # u1

    return tl, tr, bl, br
end

# function trg_tsvd!(sval, t, i; kwargs...)
#     if iseven(i)
#         p1 = (3, 4)
#         p2 = (1, 2)
#     else
#         p1 = (4, 1)
#         p2 = (2, 3)
#     end
#
#     u, s, v, err = tsvd(t, p1, p2; kwargs...)
#     @debug "tsvd error:" err
#
#     # Error is the change in singular values
#     err = norm(s - sval)
#
#     # Update the sval with the new singular value matrix now that error has been 
#     # calculated
#     copy!(sval, s)
#
#     # Calculate the SVDs
#     svd = (u=u * sqrt(s), s=s, v=sqrt(s) * v)
#
#     return svd, err
# end

function trg_decompose!(svals, cell; kwargs...)
    nx, ny = size(cell)

    Nx = lcm(2, nx)
    Ny = lcm(2, ny)

    inds = Iterators.product(1:Nx, 1:Ny)

    convs = similar(cell, Float64)

    svds = map(inds) do (x, y)
        i = x + y

        t = cell[x, y]
        sval = svals[x, y]

        if iseven(i)
            p1 = (3, 4)
            p2 = (1, 2)
        else
            p1 = (4, 1)
            p2 = (2, 3)
        end

        u, s, v, err = tsvd(t, p1, p2; kwargs...)
        @debug "tsvd error:" err

        if space(s) == space(sval)
            convs[x, y] = norm(s - sval)
        else
            convs[x, y] = Inf
        end

        # Update the sval with the new singular value matrix now that error has been 
        # calculated
        svals[x, y] = s

        # Calculate the SVDs
        svd = (u=u * sqrt(s), s=s, v=sqrt(s) * v)

        return svd
    end

    return UnitCell(svds), convs
end

function trg_contract!(cell, svds, ind; ramp=true)
    t = cell[ind]

    v1, v2, u2, u1 = trg_unitaries(svds, ind)

    @tensoropt t[1 2 3 4] = v1[3; n w] * v2[4 e n] * u1[s e; 1] * u2[w s; 2]

    return t
end

function trg_contract!(cell, svds)
    inds = CartesianIndices(cell)

    for ind in inds
        v1, v2, u2, u1 = trg_unitaries(svds, ind)

        d1 = domain(u1)
        d2 = domain(u2)
        c1 = codomain(v1)
        c2 = codomain(v2)

        # expanded_domain = d1 * d2 * c1' * c2'
        expanded_domain = d1 * d2 * c1' * c2'

        if expanded_domain == domain(cell[ind])
            t = cell[ind]
        else
            t = similar(v1, one(expanded_domain), expanded_domain)
        end
        @tensoropt t[1 2 3 4] = v1[3; n w] * v2[4 e n] * u1[s e; 1] * u2[w s; 2]

        cell[ind] = t
    end

    return cell
end

# function trg_decompose!(svals, cell::AbstractUnitCell; kwargs...)
#     nx, ny = size(cell)
#
#     Nx = lcm(2, nx)
#     Ny = lcm(2, ny)
#
#     inds = Iterators.product(1:Nx, 1:Ny)
#
#     all_convs = similar(cell, Float64)
#
#     svds = map(inds) do (x, y)
#         s = svals[x, y]
#         t = cell[x, y]
#         svd, conv = trg_tsvd!(s, t, x + y; kwargs...)
#
#         all_convs[x, y] = conv
#
#         return svd
#     end
#
#     return UnitCell(svds), all_convs
# end

function trgstep!(runtime::TRGRuntime, trunc; ramping=true)
    cell = runtime.tensors
    svals = runtime.svals

    svds, allconv = trg_decompose!(svals, cell; trunc=trunc)

    cell = trg_contract!(cell, svds)

    traces = broadcast(cell) do t
        @tensor rv = t[1 2 1 2]
        return rv
    end

    return traces, allconv
end

function step!(problem::RenormalizationProblem{<:TRG})
    runtime = problem.runtime
    alg = problem.alg

    p = 1 + problem.info.iterations

    traces, allconv = trgstep!(runtime, alg.trunc)

    if any(x -> isapprox(x, zero(x)), traces)
        # Check if it does blow up, and end if so throw a more informative error
        # than a LAPACK exception.
        throw(ErrorException("TRG failed to converge to a trace-positive solution"))
    else
        @. rmul!(runtime.tensors, 1 / traces)

        @. runtime.cumsum += log(traces) * 2.0^(-p)
    end

    # Measure of convergence is the maximum value of all the changes in the singualr values
    # obtained when truncating.
    conv = maximum(allconv)

    return conv
end
