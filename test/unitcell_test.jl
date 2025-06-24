@testsetup module SetupUnitCell
using Reexport
@reexport using TensorKit, TensorRenormalizationGroups, TestExtras, CircularArrays
end

@testitem "UnitCell" setup = [SetupUnitCell] begin
    TF = UnitCell{Square,Float64,Matrix{Float64}}
    TC = UnitCell{Square,ComplexF64,Matrix{ComplexF64}}

    matf = rand(3, 4)
    matc = rand(ComplexF64, 3, 4)

    ucf = UnitCell(matf)
    ucc = UnitCell(matc)
    @testset "Comparisons" begin
        @test size(ucf) == size(matf)
        @test ucf == matf
        @test ucf == ucf.data
    end
    @testset "Circular indexing" begin
        @test ucf[1, 1] == matf[1, 1]
        @test ucf[4, 5] == matf[1, 1]
        @test ucf[4, :] == matf[1, :]
        @test ucf[:, 5] == matf[:, 1]
        @test ucf[4:6, 5:8] == matf
        @test ucf[4, 1:8] == [matf[1, :]..., matf[1, :]...]
    end
    @testset "Similar" begin
        @test isa(similar(ucf), TF)
        @test isa(similar(ucf, ComplexF64), TC)

        new_size = (2, 2)

        sim_ucf = similar(ucf, new_size)

        @test isa(sim_ucf, TF)
        @test size(sim_ucf) == new_size

        sim_ucc = similar(ucf, ComplexF64, new_size)

        @test isa(sim_ucc, TC)
        @test size(sim_ucc) == new_size
    end
    @testset "Broadcasting" begin
        @test isa(sqrt.(ucf), TF)
        @test isa(sqrt.(ucc), TC)
        @test isa(ucc .* matf, TC)
    end
    @testset "Misc" begin
        @test isa(ITC.getdata(ucf), CircularArray)
        @test ITC.datatype(TF) == Matrix{Float64}
    end
end
