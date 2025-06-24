@testsetup module SetupMisc
using Reexport
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras, CircularArrays
end

@testitem "Other tests" setup = [SetupMisc] begin
    d = ℂ^1

    s2 = ℂ^2
    s3 = ℂ^3
    s4 = ℂ^4
    s5 = ℂ^5

    t = TensorMap(rand, ComplexF64, s2 * s3, s4 * s5)

    dom = s2 * s3 * s4' * s5'

    p1 = TensorMap(rand, ComplexF64, one(d), dom)
    p2 = TensorMap(rand, ComplexF64, d, dom)
    p3 = TensorMap(rand, ComplexF64, d * d', dom)

    tp = CompositeTensor(p3, p3')

    test_t = (p1, p2, p3)

    @testset "Utils" verbose = true begin
        @test space(ITC._transpose(t)) == ((s4' * s5') ← (s2' * s3'))
        @test @constinferred(convert(Array, ITC._transpose(t))) ==
            permutedims(convert(Array, t), (3, 4, 1, 2))
        @test @constinferred(ITC.permutedom(t, (2, 1))) == permute(t, (1, 2), (4, 3))
        @test @constinferred(ITC.permutecod(t, (2, 1))) == permute(t, (2, 1), (3, 4))

        perm = [1:10...]

        @test isa(@constinferred(ITC.tcircshift(Tuple(perm), 0)), NTuple{10,Int})

        for i in (-1, 0, 1)
            @test ITC.tcircshift(Tuple(perm), i) == Tuple(circshift(perm, i))
        end

        for tn in test_t
            @test @constinferred(ITC.rotate(tn, 0)) == tn
            @test domain(ITC.rotate(tn, 1)) == s5' * s2 * s3 * s4'
        end
        @test @constinferred(ITC.rotate(tp, 0)) == tp
        @test domain(ITC.rotate(tp, 1)[1]) == s5' * s2 * s3 * s4'
        @test codomain(ITC.rotate(tp, 1)[2]) == s5' * s2 * s3 * s4'
    end
end
