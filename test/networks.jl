@testset verbose = true begin
    e = ℂ^2
    s = ℂ^3
    w = ℂ^4
    n = ℂ^5

    p = ℂ^6

    t = TensorMap(rand, p * p', e * s * w' * n')

    @test swapaxes(t) == permute(t, ((1, 2), (4, 3, 6, 5)))
    @test invertaxes(t) == permute(t, ((1, 2), (5, 6, 3, 4)))

    ct = CompositeTensor(t, t')

    @test ct[1] == ct[2]'
    @test @constinferred(first(ct)) === parent(@constinferred(last(ct)))

    @test @constinferred(copy(ct)) == ct
    @test !(copy(ct) === ct)

    @test virtualspace(ct, 1) == e * e'
    @test virtualspace(ct, 2) == s * s'
    @test virtualspace(ct, 3) == w' * w
    @test virtualspace(ct, 4) == n' * n

    @testset "SquareSymmetric" verbose = true begin
        data = [
            1 2 3
            2 3 1
            3 1 2
        ]

        A = Matrix{Vector{Float64}}(undef, 2, 2)

        A[1, 1] = A[2, 2] = [2.0]
        A[2, 1] = A[1, 2] = [3.0]

        ucsym = UnitCell{SquareSymmetric}([1, 2, 3])

        @test UnitCell{Square}(data) == ucsym

        @test ucsym == UnitCell{SquareSymmetric}([1 2 3])
        @test ucsym == UnitCell{SquareSymmetric}([1; 2; 3;;])

        for i in 1:3
            @test ucsym[i, :] == data[i, :]
            @test ucsym[:, i] == data[:, i]
            @test ucsym[i, :] == ucsym[:, i]

            @test ucsym[i, :] isa CircularVector
            @test ucsym[:, i] isa CircularVector

            @test ucsym[i, i] isa Int
        end

        @test ucsym[1:2, 1] isa CircularVector
        @test ucsym[1:3, 1:2] isa CircularMatrix

        @test ucsym[1:3, 1:3] isa CircularMatrix

        @test ucsym[CartesianIndex(1, 2)] == ucsym[1, 2]
        @test ucsym[CartesianIndices((3, 1))] == ucsym[1:3, 1]
        @test ucsym[CartesianIndices((2, 3))] == ucsym[1:2, 1:3]

        @test identity.(ucsym) == ucsym
        @test exp2.(ucsym) == UnitCell{SquareSymmetric}([2, 4, 8])
        @test ucsym .+ ucsym == 2 * ucsym
        @test ucsym + ucsym == 2 * ucsym

        let rv1 = ucsym + data, rv2 = ucsym .+ data
            for rv in (rv1, rv2)
                @test rv == 2 * data
                @test isa(rv, UnitCell{Square})
            end
        end

        @test ucsym == ucsym'

        @test all(rmul!(copy(ucsym), 2) .== 2 * data)
        @test all(rmul!.(UnitCell{SquareSymmetric}(deepcopy(A)), 2) .== 2 * A)

        @test (zeros(Int, 3, 3) .= ucsym) == data

        let uc = UnitCell{SquareSymmetric}([1 2; 2 1])
            uc[1, 1] = 3

            @test uc[1, 1] == 3
            @test uc[2, 2] == 3

            @test uc[1, 2] == 2
            @test uc[2, 1] == 2
        end
        let uc = UnitCell{SquareSymmetric}([1 2; 2 1])
            uc[1, 2] = 3

            @test uc[1, 1] == 1
            @test uc[2, 2] == 1

            @test uc[1, 2] == 3
            @test uc[2, 1] == 3
        end
    end
end
