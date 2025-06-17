"""
    CTMRG{SVD<:OrthogonalFactorizationAlgorithm}

# Fields
- `bonddim::Int`: the bond dimension of the boundary
- `maxiter::Int = 100`: maximum number of iterations
- `tol::Float64 = 1e-12`: convergence tolerance
- `verbose::Bool = true`: when true, will print algorithm convergence progress
- `ptol::Float64 = 1e-7`: tolerance used in the pseudoinverse
- `svd_alg::SVD = TensorKit.SVD()`: algorithm used for the SVD. Either `TensorKit.SVD()` or `TensorKit.SDD()`
"""
@kwdef struct CTMRG{SVD<:OrthogonalFactorizationAlgorithm} <: AbstractCornerMethod
    bonddim::Int
    maxiter::Int = 100
    tol::Float64 = 1e-12
    verbose::Bool = true
    ptol::Float64 = 5e-8
    svdalg::SVD = TensorKit.SVD()
    randinit::Bool = false
end

function step!(runtime::CornerMethodRuntime, algorithm::CTMRG)
    primary_tensors = runtime.primary
    permuted_tensors = runtime.permuted

    bonddim = algorithm.bonddim

    ctmrgmove!(primary_tensors, bonddim; svd_alg=algorithm.svdalg, ptol=algorithm.ptol)

    # Write updated tensors into the permuted placeholders
    updatecorners!(permuted_tensors, primary_tensors)

    # Sweep along the y axis (up/down)
    ctmrgmove!(permuted_tensors, bonddim; svd_alg=algorithm.svdalg, ptol=algorithm.ptol)

    # Update the unpermuted (primary) tensors.
    updatecorners!(primary_tensors, permuted_tensors)

    # error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)
    error = ctmerror!(runtime)

    return error
end

## PROJECTORS

function projectors(
    C1_00,
    C2_30,
    C3_33,
    C4_03,
    T1_10,
    T1_20,
    T2_31,
    T2_32,
    T3_13,
    T3_23,
    T4_01,
    T4_02,
    M_11,
    M_21,
    M_12,
    M_22,
    bonddim;
    svd_alg=TensorKit.SVD(),
    kwargs...,
)
    # Top
    top = halfcontract(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    # println(top)
    U, S, V = tsvd!(normalize!(top); alg=svd_alg)
    FUL = sqrt(S) * V
    FUR = U * sqrt(S)

    # Bottom (requires permutation)
    MP_12 = invertaxes(M_12)
    MP_22 = invertaxes(M_22)

    bot = halfcontract(C3_33, T3_23, T3_13, C4_03, T2_32, MP_22, MP_12, T4_02)
    U, S, V = tsvd!(normalize!(bot); alg=svd_alg)
    FDL = U * sqrt(S)
    FDR = sqrt(S) * V

    UL, VL = biorth_truncation(FUL, FDL, bonddim; svd_alg=svd_alg, kwargs...)
    VR, UR = biorth_truncation(FDR, FUR, bonddim; svd_alg=svd_alg, kwargs...)

    return UL, VL, UR, VR
end

function biorth_truncation(U0, V0, xi; ptol=5e-8, svd_alg=TensorKit.SVD())
    Q, S, W = tsvd!(U0 * V0; trunc=truncdim(xi), alg=svd_alg)

    # normalize!(S)

    SX = pinv(sqrt(S); rtol=ptol)

    # normalize!(SX)
    U_temp = SX * Q' * U0
    U = _transpose(U_temp)

    V = V0 * W' * SX

    # biorthness = norm(normalize(U_temp * V) - normalize(one(U_temp * V)))
    #
    # btol = sqrt(eps())
    #
    # if biorthness > btol
    #     @warn "Biorthogonalisation failed!" biorthness
    #     @info "" U_temp * V
    # end
    # normalize!(U)
    # normalize!(V)

    return U, V
end

##

function ctmrgmove!(ctmrg::CornerMethodTensors, bonddim; kwargs...)
    C1, C2, C3, C4 = ctmrg.corners
    T1, T2, T3, T4 = ctmrg.edges
    UL, VL, UR, VR = ctmrg.projectors

    network = ctmrg.network

    if geometrytype(typeof(network)) == SquareSymmetric

        # xax should be the short axes in this case

        xax = axes(eachindex(network), 2)
        yax = axes(eachindex(network), 1)

        C1o, C2o, C3o, C4o = map(copy, ctmrg.corners)
        T1o, T2o, T3o, T4o = map(copy, ctmrg.edges)

    else
        xax = axes(network, 1)
        yax = axes(network, 2)

        C1o, C2o, C3o, C4o = ctmrg.corners
        T1o, T2o, T3o, T4o = ctmrg.edges
    end

    for x in xax
        # Compute all the unique projectors
        for y in yax
            UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = projectors(
                C1[x + 0, y + 0],
                C2[x + 3, y + 0],
                C3[x + 3, y + 3],
                C4[x + 0, y + 3],
                T1[x + 1, y + 0],
                T1[x + 2, y + 0],
                T2[x + 3, y + 1],
                T2[x + 3, y + 2],
                T3[x + 1, y + 3],
                T3[x + 2, y + 3],
                T4[x + 0, y + 1],
                T4[x + 0, y + 2],
                network[x + 1, y + 1],
                network[x + 2, y + 1],
                network[x + 1, y + 2],
                network[x + 2, y + 2],
                bonddim;
                kwargs...,
            )
        end

        for y in yax
            #=
              1
            C --- T ---
            |2    |
            *--V--*
               |
            =#
            # projectcorner!(
            #     C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
            # )
            C1[x + 1, y + 0] = projectcorner(
                C1o[x + 0, y + 0], T1o[x + 1, y + 0], VL[x + 0, y + 0]
            )
            #=
                    1
            --- T --- C
                |    2|
                *--V--*
                   |
            =#
            # projectcorner!(
            #     C2[x + 2, y + 0],
            #     C2[x + 3, y + 0],
            #     swapvirtual(T1[x + 2, y + 0]),
            #     VR[x + 3, y + 0],
            # )
            C2[x + 2, y + 0] = projectcorner(
                C2o[x + 3, y + 0], swapvirtual(T1o[x + 2, y + 0]), VR[x + 3, y + 0]
            )

            # projectcorner!(
            #     C3[x + 2, y + 3], C3[x + 3, y + 3], T3[x + 2, y + 3], UR[x + 3, y + 2]
            # )
            C3[x + 2, y + 3] = projectcorner(
                C3o[x + 3, y + 3], T3o[x + 2, y + 3], UR[x + 3, y + 2]
            )
            # projectcorner!(
            #     C4[x + 1, y + 3],
            #     C4[x + 0, y + 3],
            #     swapvirtual(T3[x + 1, y + 3]),
            #     UL[x + 0, y + 2],
            # )
            C4[x + 1, y + 3] = projectcorner(
                C4o[x + 0, y + 3], swapvirtual(T3o[x + 1, y + 3]), UL[x + 0, y + 2]
            )
            #=
               |
            *--U--*
            |2    |
            T --- M ---
            |1    |
            *--V--*
               |
            =#
            # projectedge!(
            #     T4[x + 1, y + 1],
            #     T4[x + 0, y + 1],
            #     network[x + 1, y + 1],
            #     UL[x + 0, y + 0],
            #     VL[x + 0, y + 1],
            # )
            T4[x + 1, y + 1] = projectedge(
                T4o[x + 0, y + 1], network[x + 1, y + 1], UL[x + 0, y + 0], VL[x + 0, y + 1]
            )
            #=
                   |
                *--U--*
                |    1|
            --- M --- T
                |    2|
                *--V--*
                   |
            =#
            # projectedge!(
            #     T2[x + 2, y + 1],
            #     T2[x + 3, y + 1],
            #     invertaxes(network[x + 2, y + 1]),
            #     # flipaxis(network[x + 2, y + 1]),
            #     VR[x + 3, y + 1],
            #     UR[x + 3, y + 0],
            # )
            T2[x + 2, y + 1] = projectedge(
                T2o[x + 3, y + 1],
                invertaxes(network[x + 2, y + 1]),
                # flipaxis(network[x + 2, y + 1]),
                VR[x + 3, y + 1],
                UR[x + 3, y + 0],
            )
        end
    end

    # @info "corner" norm.(ctmrg.corners)
    # @info "edge" norm.(ctmrg.edges)

    normalize!(ctmrg.corners)
    normalize!(ctmrg.edges)

    # foreach(c -> println(space(c[1,1])),ctmrg.corners)

    return ctmrg
end

swapvirtual(t::AbstractTensorMap) = permutedom(t, (2, 1))

# function flipaxis(t::TensorPair)
#     return TensorPair(permutedom(t.top, (3, 2, 1, 4)), permutedom(t.bot, (3, 2, 1, 4)))
# end

function contract(ctmrg::CornerMethodRuntime, network, i1::UnitRange, i2::UnitRange)
    return contract(ctmrg.primary, network, i1, i2)
end
function contract(ctmrg::CornerMethodTensors, network, i1::UnitRange, i2::UnitRange)
    cs = getboundary(ctmrg.corners, i1, i2)
    es = map(x -> tuple(x...), getboundary(ctmrg.edges, i1, i2))
    return _contractall(cs..., es..., network)
end

function testctmrg(data_func; T=Float64, D=10)
    βc = log(1 + sqrt(2)) / 2

    s = ℂ^2

    N = 4

    network =
        x -> UnitCell{SquareSymmetric}(
            fill(TensorMap(T.(data_func(x)[1]), one(s), s * s * s' * s'), N, N)
        )
    network_magn =
        x -> UnitCell{SquareSymmetric}(
            fill(TensorMap(T.(data_func(x)[2]), one(s), s * s * s' * s'), N, N)
        )

    rv = []
    rv_exact = []

    alg = CTMRG(;
        bonddim=D,
        verbose=true,
        maxiter=1000,
        tol=1e-10,
        svdalg=TensorKit.SVD(),
        randinit=false,
    )
    # alg = FPCM(; bonddim=D, verbose=true, maxiter=500, tol=1e-11, randinit=false)
    #
    # # alg = algf
    # alg = HybridCornerMethod(;
    #     bonddim=10,
    #     maxiter=100,
    #     randinit=false,
    #     ctmrg=algc,
    #     fpcm=algf,
    #     dofpcm=AtFrequency(5),
    # )
    # alg = VUMPS(; bonddim=D, verbose=true, maxiter=200)

    # randgauge! =
    #     (p, X, Y) -> @tensoropt p[a b c d] =
    #         copy(p)[aa bb cc dd] * X[aa; a] * inv(X)[c; cc] * Y[bb; b] * inv(Y)[d; dd]

    for x in (1.1,)
        b1 = network(x * βc)
        b2 = network_magn(x * βc)

        for y in axes(b1, 2)
            for x in axes(b1, 1)
                A, _, _ = tsvd(randn(T, s, s))
                randgauge!(b1, x, y, one(A) + 1e-1 * A)
                randgauge!(b2, x, y, one(A) + 1e-1 * A)
            end
        end

        # b1 = randgauge!.(network(x * βc), Ref(X), Ref(Y))
        # b2 = randgauge!.(network_magn(x * βc), Ref(X), Ref(Y))
        if x < 1.0
            M = 0.0
        else
            M = abs((1 - sinh(2 * x * βc)^(-4))^(1 / 8))
        end

        cb = (st, args...) -> println(contract(st.tensors, b2) ./ contract(st.tensors, b1))

        state = newrenormalization(alg, b1)#; callback=cb)

        did_converge = false

        renormalize!(state)

        Z = contract(state.runtime, b1)
        magn = contract(state.runtime, b2) ./ Z

        push!(rv, abs.(magn)[1, 1])

        println(abs.(magn))

        push!(rv_exact, M)
    end

    @info rv

    return abs.(rv - rv_exact)
end

function randgauge!(net, x, y, X)
    p1 = net[x, y]
    p2 = net[x + 1, y]

    @tensoropt p1[a b c d] = copy(p1)[aa b c d] * X[aa; a]
    @tensoropt p2[a b c d] = copy(p2)[a b cc d] * inv(X)[c; cc]

    return nothing
end

function toric(p)
    distr = Dict("X" => p / 3, "Y" => p / 3, "Z" => p / 3, "I" => 1 - p)

    function xprob(dst)
        arr = zeros(2, 2, 2, 2)
        arr[1, 1, 1, 1] = arr[1, 2, 1, 2] = arr[2, 1, 2, 1] = arr[2, 2, 2, 2] = dst["X"]
        arr[1, 2, 1, 1] = arr[1, 1, 1, 2] = arr[2, 2, 2, 1] = arr[2, 1, 2, 2] = dst["I"]
        arr[2, 1, 1, 1] = arr[1, 1, 2, 1] = arr[2, 2, 1, 2] = arr[1, 2, 2, 2] = dst["Y"]
        arr[2, 1, 1, 2] = arr[1, 2, 2, 1] = arr[2, 2, 1, 1] = arr[1, 1, 2, 2] = dst["Z"]
        return arr
    end

    data1 = xprob(distr)
    data2 = permutedims(data1, (2, 1, 4, 3))

    kron = delta(Float64, 2, 2, 2, 2)

    s = ℂ^2

    toten = x -> TensorMap(x, one(s), s * s * s' * s')

    kront = toten(kron)
    ten1 = toten(data1)
    ten2 = toten(data2)

    # bulk = UnitCell([ten1 kront ten1; kront ten2 kront; ten1 kront ten1])
    bulk = UnitCell(
        [
            ten1 kront ten1 kront
            kront ten2 kront ten2
            ten1 kront ten1 kront
            kront ten2 kront ten2
        ],
    )
    # bulk = UnitCell([ten1 kront; kront ten2])

    alg = CTMRG(;
        bonddim=50,
        verbose=true,
        maxiter=500,
        tol=1e-10,
        svdalg=TensorKit.SVD(),
        randinit=false,
    )

    state = initialize(bulk, alg)

    calculate!(state)
    out = contract(state.tensors, bulk)

    return out
end

function dimer(D; el=Float64)
    rv = []

    probs = []
    # TS = 0.5:0.01:0.8

    TS = 0.8:0.8

    for T in TS
        β = 1 / T

        a = zeros(el, 4, 4, 4, 4)
        b = zeros(el, 4, 4, 4, 4)

        for ind in CartesianIndices(a)
            if ind[1] ==
                mod(ind[2] + 1, 1:4) ==
                mod(ind[3] + 2, 1:4) ==
                mod(ind[4] + 3, 1:4)
                a[ind] = 1
            end
            if ind[1] ==
                mod(ind[2] - 1, 1:4) ==
                mod(ind[3] - 2, 1:4) ==
                mod(ind[4] - 3, 1:4)
                b[ind] = 1
            end
        end

        q = sqrt([1 0 0 0; 0 exp(β / 2) 1 1; 0 1 1 1; 0 1 1 exp(β / 2)])

        qh = diagm([1, -1, 1, -1]) * q
        qv = diagm([-1, 1, -1, 1]) * q

        @tensoropt aa[ii, jj, kk, ll] :=
            a[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]
        @tensoropt bb[ii, jj, kk, ll] :=
            b[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]

        δa = zeros(4, 4)
        δa[1, 1] = δa[3, 3] = 1
        δa[2, 2] = δa[4, 4] = -1

        @tensoropt aad[i, j, k, l] := a[ii, j, k, l] * δa[i, ii]
        @tensoropt bbd[i, j, k, l] := b[ii, j, k, l] * δa[i, ii]

        s = ℂ^4

        A = TensorMap(aa, one(s), s * s * s' * s')
        B = TensorMap(bb, one(s), s * s * s' * s')

        AD = TensorMap(aad, one(s), s * s * s' * s')
        BD = TensorMap(bbd, one(s), s * s * s' * s')

        alg = CTMRG(; bonddim=D, maxiter=149, tol=1e-11, randinit=false)
        # alg = VUMPS(; bonddim=D, maxiter=1000)

        bulk = UnitCell([A B; B A])
        dbulk = UnitCell([AD BD; BD AD])

        st = initialize(bulk, alg)

        sto = calculate(st)
        d = (contract(sto.tensors, dbulk) ./ contract(sto.tensors, bulk)) .* 4

        ps = Dict{Int,Matrix{Float64}}([])

        csum = zeros(2, 2)

        for l in 1:4
            δp = zeros(4, 4)
            δp[l, l] = 1

            @tensoropt aap[i, j, k, l] := a[ii, j, k, l] * δp[i, ii]
            @tensoropt bbp[i, j, k, l] := b[ii, j, k, l] * δp[i, ii]

            AP = TensorMap(aap, one(s), s * s * s' * s')
            BP = TensorMap(bbp, one(s), s * s * s' * s')

            pbulk = UnitCell([AP BP; BP AP])

            p = (contract(sto.tensors, pbulk) ./ contract(sto.tensors, bulk))

            ps[l] = abs.(p)

            csum = csum + ps[l]
        end

        for l in 1:4
            ps[l] = ps[l] ./ csum
        end
        push!(probs, ps)
        push!(rv, abs(d[1, 1]))
    end
    return rv, probs
end
