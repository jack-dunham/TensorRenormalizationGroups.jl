@testsetup module SetupCTMRG
using Reexport
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras, CircularArrays
end

@testitem "Corner Methods" setup = [SetupCTMRG] begin
    s = ℂ^3
    suc = UnitCell([s;;])

    tf = s -> TensorMap(rand, ComplexF64, one(s), s * s)

    C1 = map(s -> tf(s), suc)
    C2 = map(s -> tf(s'), suc)
    C3 = map(s -> tf(s), suc)
    C4 = map(s -> tf(s'), suc)

    corners = Corners((C1, C2, C3, C4))

    @test @constinferred(Corners(C1, C2, C3, C4)) == corners

    CT = typeof(C1)

    @test length(corners) == 4

    @test iterate(corners, 5) === nothing
    @test convert(Tuple, corners) == (C1, C2, C3, C4)
    @test convert(Tuple, (C1, C2, C3, C4)) == (C1, C2, C3, C4)

    @test isa(identity.(corners), NTuple{4,CT})

    @test isa(@constinferred(convert(Tuple, corners)), NTuple{4,CT})

    @test isa(@constinferred(map(identity, corners)), NTuple{4,CT})

    @test scalartype(corners) == ComplexF64

    @test ITC.chispace(corners) == s
end

@testitem "CTMRG" setup = [SetupCTMRG] begin
    d = ℂ^2
    s1 = ℂ^3
    s2 = ℂ^4
    chi = ℂ^5

    ta = TensorMap(rand, ComplexF64, d * d', s1 * s1 * s2' * s2')
    tb = TensorMap(rand, ComplexF64, d * d', s2 * s2 * s1' * s1')

    net = broadcast(UnitCell([ta tb; tb ta])) do t
        return CompositeTensor(t, t')
    end

    alg = CTMRG(; verbose=false, bonddim=1, randinit=false)

    uc = ITC.ensure_contractable(net)

    @testset "Initialization" verbose = true begin
        ten = @constinferred(ITC.inittensors(uc, alg))

        st = @constinferred(initialize(uc, alg))

        prob = @constinferred(Renormalization(uc, alg, st))

        @test eltype(prob.network) <: CompositeTensor

        for sp in convert.(ProductSpace, domain(ta))
            for ch in convert.(ProductSpace, (ℂ^2, sp, ℂ^6))
                eiso = @constinferred(ITC.get_embedding_isometry(sp, ch))

                @test codomain(eiso) == sp
                @test domain(eiso) == ch
            end
            riso = @constinferred(ITC.get_removal_isometry(sp))

            @test codomain(riso) == sp
            @test domain(riso) == one(sp)
        end

        c_raw = @constinferred(ITC.initcorners(uc, chi))
        e_raw = @constinferred(ITC.initedges(uc, chi))

        @constinferred(ITC.randomize_if_zero!(c_raw))
        @constinferred(ITC.randomize_if_zero!(e_raw))

        C1, C2, C3, C4 = c_raw
        T1, T2, T3, T4 = c_raw

        @constinferred(Corners((C1, C2, C4, C4)))
        @constinferred(Corners(C1, C2, C3, C4))

        @constinferred(Edges((T1, T2, T4, T4)))
        @constinferred(Edges(T1, T2, T3, T4))

        T1, T2, T3, T4 = e_raw

        for y in axes(uc, 2)
            for x in axes(uc, 1)
                @test codomain(T1[x, y]) == virtualspace(uc[x, y + 1], 4)
                @test codomain(T2[x, y]) == virtualspace(uc[x - 1, y], 1)
                @test codomain(T3[x, y]) == virtualspace(uc[x, y - 1], 2)
                @test codomain(T4[x, y]) == virtualspace(uc[x + 1, y], 3)
            end
        end

        for ind in eachindex(C1)
            @test domain(C1[ind]) == chi * chi
            @test domain(C2[ind]) == chi' * chi'
            @test domain(C3[ind]) == chi * chi
            @test domain(C4[ind]) == chi' * chi'
        end
    end

    @testset "Error calculation" verbose = true begin
        corn = ITC.initcorners(uc, chi)
        state = initialize(uc, alg)

        Ss = @constinferred(ITC.initerror(state.primary))
        Ss_old = deepcopy(Ss)

        @test isa(@constinferred(ITC.ctmerror!(Ss, corn)), AbstractFloat)
        # Singular values should have updated:
        @test Ss !== Ss_old
        # Should get zero error with corners unchanged:
        C1, C2, C3, C4 = corn
        @test @constinferred(ITC.boundaryerror!(Ss[1], C1)) ≈ zero.(eltype.(C1))

    end
end
