@testsetup module SetupIsing
using Random
using Reexport
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras

Random.seed!(1234)

import TensorRenormalizationGroups.KrylovKit.initialize

export classicalisingmpo, exactZ, isingnetwork, initialize

function statmechmpo(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo(β; J=1.0, h=0.0)
    return statmechmpo(β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2)
end

function exactZ(β)
    function quad_midpoint(f, a, b, N)
        h = (b - a) / N
        int = 0.0
        for k in 1:N
            xk_mid = (b - a) * (2k - 1) / (2N) + a
            int = int + h * f(xk_mid)
        end
        return int
    end

    function exact_integrand(x, β)
        return log(
            cosh(2 * β)^2 +
            sinh(2 * β) * sqrt(sinh(2 * β)^2 + sinh(2 * β)^(-2) - 2 * cos(2 * x)),
        )
    end
    f = x -> exact_integrand(x, β)
    quad_out = quad_midpoint(f, 0, pi, 100)
    return 1 / (2 * pi) * quad_out + log(2) / 2
end

function isingnetwork(::Type{G}, ::Type{T}, β, N) where {G,T}
    s = ComplexSpace(2)

    Z, M = map(classicalisingmpo(β)) do data
        tensor = TensorMap(T.(data), one(s), s * s * s' * s')
        return UnitCell{G}(fill(tensor, N, N))
    end

    for y in axes(Z, 2)
        for x in axes(Z, 1)
            A, _, _ = tsvd(randn(T, s, s))
            randgauge!(Z, x, y, one(A) + 1.0e-1 * A)
            randgauge!(M, x, y, one(A) + 1.0e-1 * A)
        end
    end

    return Z, M
end

function randgauge!(net, x, y, X)
    p1 = net[x, y]
    p2 = net[x + 1, y]

    @tensoropt p1[a b c d] = copy(p1)[aa b c d] * X[aa; a]
    @tensoropt p2[a b c d] = copy(p2)[a b cc d] * inv(X)[c; cc]

    return nothing
end

end # SetupIsing

@testsetup module SetupIsingPEPS
using Reexport
using Random
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras

export CZ1L

function testpeps(x)
    βc = log(1 + sqrt(2)) / 2
    β = x * log(1 + sqrt(2)) / 2

    exact_M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

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

    alg = VUMPS(; maxiter=100, bonddim=50)
    # alg = CTMRG(; maxiter = 1000, tol = 1.0e-10, svdalg = TensorKit.SVD(), bonddim = 50)
    rt = Renormalization(net, alg)

    renormalize!(rt)

    rv = []

    for l in 2:10
        bulk = fill(hsm, l - 2)

        magn = contract([hsmZ, bulk..., hsmZ], rt.runtime, 1:l, 1:1)
        magn_norm = contract([hsm, bulk..., hsm], rt.runtime, 1:l, 1:1)

        push!(rv, magn / magn_norm)
    end

    return rv
end

const CZ1L = testpeps(1.1)

end # SetupIsingPEPS

@testitem "Classical Ising model" setup = [SetupIsing] begin
    βc = log(1 + sqrt(2)) / 2
    x = 1.1
    M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

    algs = (
        VUMPS(; verbose=true, maxiter=1000, tol=1.0e-10, bonddim=20),
        CTMRG(;
            verbose=true, maxiter=1000, tol=1.0e-10, svdalg=TensorKit.SVD(), bonddim=20
        ),
    )

    @testset "Geometry $G" for G in (Square, SquareSymmetric)
        @testset "Unit cell size $N" for N in (1, 2, 3)
            Znet, Mnet = isingnetwork(G, ComplexF64, x * βc, N)

            @testset "Using $(typeof(alg))" for alg in algs
                runtime = @constinferred(initialize(Znet, alg))
                state = @constinferred(Renormalization(Znet, alg, runtime))

                renormalize!(state)

                z_val = @constinferred(contract(Znet, state))
                m_val = contract(Mnet, state) ./ z_val

                for y in 1:N
                    for x in 1:N
                        @test abs(m_val[x, y]) ≈ M atol = 1.0e-10
                    end
                end
            end

            trgalg = TRG(; verbose=false, maxiter=100, trunc=TensorKit.truncdim(10))

            z_exact = exactZ(βc * x)

            @testset "Using $(typeof(trgalg))" begin
                Znet, Mnet = isingnetwork(G, Float64, x * βc, N)

                rt = @constinferred(initialize(Znet, trgalg))
                st = @constinferred(Renormalization(Mnet, trgalg, rt))

                renormalize!(st)

                z_val = st.runtime.cumsum

                for y in 1:N
                    for x in 1:N
                        @test abs(z_val[x, y]) ≈ z_exact atol = 1.0e-4
                    end
                end
            end
        end
    end
end

@testitem "Classical Ising Model (PEPS)" setup = [SetupIsingPEPS] begin
    x = 1.1
    βc = log(1 + sqrt(2)) / 2
    β = x * βc
    exact_M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    op = exp(β / 2 * Z ⊗ Z)

    U, s, V = tsvd(op, (1, 3), (2, 4); trunc=truncbelow(eps()))

    E = S = U * sqrt(s)
    W = N = sqrt(s) * V

    plus = Tensor(1 / sqrt(2) * [1, -1], ℂ^2)

    @tensoropt state[out; e s w n] :=
        E[out x4; e] * S[x4 x3; s] * W[w; x3 x2] * N[n; x2 x1] * plus[x1]

    function randgauge_peps!(net, x, y, X)
        p1 = net[x, y]
        p2 = net[x + 1, y]

        @tensoropt p1[out; a b c d] = copy(p1)[out; aa b c d] * X[aa; a]
        @tensoropt p2[out; a b c d] = copy(p2)[out; a b cc d] * inv(X)[c; cc]

        return nothing
    end

    size = 2

    algs = (
        VUMPS(; maxiter=1000, tol=1.0e-10, bonddim=50),
        CTMRG(; maxiter=1000, tol=1.0e-9, svdalg=TensorKit.SVD(), bonddim=50),
    )

    @testset "Geometry $G" for G in (Square, SquareSymmetric)
        net_state = UnitCell{G}(fill(state, size, size))

        for y in axes(net_state, 2)
            for x in axes(net_state, 1)
                A, _, _ = tsvd(randn(eltype(state), ComplexSpace(2), ComplexSpace(2)))
                randgauge_peps!(net_state, x, y, one(A) + 1.0e-1 * A)
            end
        end

        net = map(CompositeTensor, net_state, adjoint.(net_state))
        netZ = map(CompositeTensor, net_state, adjoint.(Ref(Z) .* net_state))

        @testset "Using $(typeof(alg))" for alg in algs
            rt = Renormalization(net, alg)

            renormalize!(rt)

            norm = contract(net, rt)[1, 1]
            magn_ss = contract(netZ, rt)[1, 1] / norm

            @test abs(magn_ss) ≈ exact_M atol = 1.0e-10

            # magn_ss = 1

            rv = []

            for l in 2:10
                bulk = fill(state, l - 2)

                ops = [Z * state, bulk..., Z * state]
                bare_ops = [state, bulk..., state]

                magn_in = map(CompositeTensor, bare_ops, adjoint.(ops))
                magn_norm_in = map(CompositeTensor, bare_ops, adjoint.(bare_ops))

                magn = contract(magn_in, rt.runtime, 1:l, 1)
                magn_norm = contract(magn_norm_in, rt.runtime, 1:l, 1)

                push!(rv, magn / magn_norm)
            end

            @test rv ≈ CZ1L
        end
    end
end
#=
    @testset "Interacting dimers" verbose = true begin
        β = 1 / 0.85

        a = zeros(4, 4, 4, 4)
        b = zeros(4, 4, 4, 4)

        for I in CartesianIndices(a)
            if I[1] == mod(I[2] + 1, 1:4) == mod(I[3] + 2, 1:4) == mod(I[4] + 3, 1:4)
                a[I] = 1
            end
            if I[1] == mod(I[2] - 1, 1:4) == mod(I[3] - 2, 1:4) == mod(I[4] - 3, 1:4)
                b[I] = 1
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

        alg = VUMPS(; bonddim=5, maxiter=1000)

        bulk = UnitCell(([A B; B A]))
        dbulk = UnitCell(([AD BD; BD AD]))

        st = initialize(bulk, alg)
        dst = initialize(dbulk, alg)

        sto = calculate(st)
        contract(sto.tensors, dbulk) ./ contract(sto.tensors, bulk)
    end
    =#

#=
@testset "Double layer" verbose = true begin
    @testset "Classical Ising model" verbose = true begin
        βc = log(1 + sqrt(2)) / 2

        x = 1.5

        M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

        D = ℂ^2

        zt, mt = map(
            data -> TensorMap(data, one(D), D * D * D' * D'), classicalisingmpo(x * βc)
        )
        zbulk = UnitCell([zt;;])
        mtbulk = UnitCell([mt;;])

        χs = [2, 5, 20]

        algs = (
            (VUMPS(; bonddim=i) for i in χs)...,
            (CTMRG(; svd_alg=TensorKit.SVD(), bonddim=i) for i in χs)...,
        )

        @testset "Using $(typeof(alg)) with χ = $(alg.bonddim)" for alg in algs
            st = @constinferred(initialize(zbulk, alg))
            calculate!(st)
            z_val = @constinferred(contract(st.tensors, zbulk))
            m_val = contract(st.tensors, mtbulk) ./ z_val

            @test isapprox(abs(m_val[1, 1]), M; atol=1e-5)
        end
    end
end
=#
