@testsetup module SetupVUMPS
using Reexport
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras, CircularArrays
end

@testitem "VUMPS" setup = [SetupVUMPS] begin
    TF = Float64
    TC = ComplexF64

    inds = CartesianIndices((1:4, 1:3)) # three matrix product states of length 4

    @testset "MPS for eltype = $T" verbose = true for T in (TF, TC)
        d = ℂ^2
        χ = ℂ^4

        ds = UnitCell(map(_ -> d, inds))
        χs = UnitCell(map(_ -> χ, inds))
        χs1 = UnitCell(map(_ -> ℂ^1, inds))

        A = UnitCell(map(_ -> TensorMap(randn, T, d, χ * χ'), inds))
        C = UnitCell(map(_ -> TensorMap(randn, T, one(d), χ * χ'), inds))

        a = A[1, 1]
        cl = C[1, 1]
        cr = C[0, 1]

        @test isa(TRGroups._similar_ac(a, cl), typeof(a))
        @test isa(TRGroups._similar_ac(cr, a), typeof(a))

        @test TRGroups.centraltensor(C, A)[1, 1] == TRGroups.mulbond(cr, a)
        @test TRGroups.centraltensor(A, C)[1, 1] == TRGroups.mulbond(a, cl)

        mps = @constinferred(MPS(A))
        @test @constinferred(scalartype(mps)) == T

        AL, C, AR, AC = mps
        @test all((AL, C, AR, AC) == TRGroups.unpack(mps))
        @testset "Mixed canonical form" begin
            let mps = MPS(rand, T, ds, χs1), C = mps.C
                AL, C, AR, AC = mps
                @test all(@. domain(C) == χs1 * adjoint(χs1))
            end

            mps = @constinferred(MPS(rand, T, ds, χs))

            physbonds = UnitCell{SquareSymmetric}(fill(ComplexSpace(2), 2, 2))
            virtbonds = UnitCell{SquareSymmetric}(fill(ComplexSpace(2), 2, 2))
            mps_sym = MPS(rand, T, physbonds, virtbonds)

            @test TRGroups.geometrytype(mps_sym.C) == SquareSymmetric

            for uc in TRGroups.unpack(mps)
                @test TRGroups.scalartype(uc) == T
            end

            @test TRGroups.isgauged(mps) == 1
            @test TRGroups.isgauged(mps_sym) == 1
            @test TRGroups.isgauged(MPS(A, C, A, A)) == 0
            @test TRGroups.isgauged(MPS(similar(A), C, A, A)) == -1
        end
    end
end
