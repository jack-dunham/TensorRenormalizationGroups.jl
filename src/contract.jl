function contract(problem::Renormalization, inds...)
    return contract(problem.network, problem, inds...)
end
function contract(network::AbstractMatrix, boundary)
    return contract(network, boundary, CartesianIndices(network))
end
function contract(network::AbstractMatrix, boundary, inds::CartesianIndices{2})
    RT = promote_type(scalartype(boundary), scalartype(network))

    rv = similar(network, RT)

    broadcast!(rv, inds) do ind
        return contract([network[ind];;], boundary, ind)
    end

    return rv
end

function contract(bulk::AbstractArray, boundary, inds...)
    return contract(bulk, boundary, to_indices(bulk, (inds))...)
end

function contract(bulk::AbstractArray, boundary::Renormalization, i1, i2)
    return contract(bulk, boundary.runtime, i1, i2)
end
# Every call to `contract` should call this method eventually.
function contract(bulk::AbstractVector, boundary::AbstractRenormalizationRuntime, i1, i2)
    bulkmat = similar(bulk, (length(i1), length(i2)))
    try
        bulkmat .= bulk
    catch e
        if e isa DimensionMismatch
            throw(
                DimensionMismatch("
                              size of provided array must coincide with the length of the 
                              indices provided
                              ")
            )
        else
            rethrow()
        end
    end
    return __contract(bulkmat, boundary, i1, i2)
end
function contract(bulk::AbstractMatrix, boundary::AbstractRenormalizationRuntime, i1, i2)
    if length(i1) != size(bulk, 1) || length(i2) != size(bulk, 2)
        throw(
            DimensionMismatch("
                              size of provided array must coincide with the length of the 
                              indices provided
                              ")
        )
    end
    return __contract(bulk, boundary, i1, i2)
end

__contract(MS, boundary, i1, i2) = _contract(MS, boundary, i1, i2)
function __contract(MS::AbstractArray{<:CompositeTensor{2}}, boundary, i1, i2)
    MSu = first.(MS)
    MSd = last.(MS)

    return _contract(MSu, MSd, boundary, i1, i2)
end

function get_bond_symbol(i, j, dir, extra="")
    if dir === :h
        str = "$(i)$(j)_$(i + 1)$(j)"
    elseif dir === :v
        str = "$(i)$(j)_$(i)$(j + 1)"
    end
    return Symbol(str * extra)
end

@generated function _contractall(
    C1,
    C2,
    C3,
    C4,
    T1::NTuple{Nx,E},
    T2::NTuple{Ny,E},
    T3::NTuple{Nx,E},
    T4::NTuple{Ny,E},
    MS::AbstractArray{<:AbsTen{0}},
) where {Nx,Ny,E}
    gh = (i, j) -> get_bond_symbol(i, j, :h)
    gv = (i, j) -> get_bond_symbol(i, j, :v)

    e_T1s = [
        Expr(:ref, Expr(:ref, :T1, i), gv(i + 1, 1), (gh(i + 1, 1), gh(i, 1))...) for
        i in 1:Nx
    ]

    e_T2s = [
        Expr(
            :ref,
            Expr(:ref, :T2, j),
            gh(Nx + 1, j + 1),
            (gv(Nx + 2, j), gv(Nx + 2, j + 1))...,
        ) for j in 1:Ny
    ]
    e_T3s = [
        Expr(
            :ref,
            Expr(:ref, :T3, i),
            gv(i + 1, Ny + 1),
            (gh(i, Ny + 2), gh(i + 1, Ny + 2))...,
        ) for i in 1:Nx
    ]
    e_T4s = [
        Expr(:ref, Expr(:ref, :T4, j), gh(1, j + 1), (gv(1, j + 1), gv(1, j))...) for
        j in 1:Ny
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i, j),
            gh(i + 1, j + 1),
            gv(i + 1, j + 1),
            gh(i, j + 1),
            gv(i + 1, j),
        ) for i in 1:Nx, j in 1:Ny
    ]

    e_C1 = Expr(:ref, :C1, gh(1, 1), gv(1, 1))
    e_C2 = Expr(:ref, :C2, gh(Nx + 1, 1), gv(Nx + 2, 1))
    e_C3 = Expr(:ref, :C3, gh(Nx + 1, Ny + 2), gv(Nx + 2, Ny + 1))
    e_C4 = Expr(:ref, :C4, gh(1, Ny + 2), gv(1, Ny + 1))

    e_einsum = Expr(
        :call, :*, e_C1, e_C2, e_C3, e_C4, e_T1s..., e_T2s..., e_T3s..., e_T4s..., e_MS...
    )

    quote
        @tensoropt rv = $e_einsum
    end
end

@generated function _contractall(
    C1,
    C2,
    C3,
    C4,
    T1::NTuple{Nx,E},
    T2::NTuple{Ny,E},
    T3::NTuple{Nx,E},
    T4::NTuple{Ny,E},
    MS::AbstractArray{<:AbsTen{NInd}},
    MSP::AbstractArray{<:TenAbs{NInd}},
) where {NInd,Nx,Ny,E}
    gh = (i, j) -> get_bond_symbol(i, j, :h)
    gv = (i, j) -> get_bond_symbol(i, j, :v)

    gha = (i, j) -> get_bond_symbol(i, j, :h, "a")
    ghb = (i, j) -> get_bond_symbol(i, j, :h, "b")

    gva = (i, j) -> get_bond_symbol(i, j, :v, "a")
    gvb = (i, j) -> get_bond_symbol(i, j, :v, "b")

    symb = (i, j, s) -> Symbol("$(s)_$(i)_$(j)")

    phys = (i, j) -> begin
        if NInd == 1
            [symb(i, j, :p)]
        elseif NInd == 2
            [symb(i, j, :pk), symb(i, j, :pb)]
        end
    end

    e_T1s = [
        Expr(
            :ref,
            Expr(:ref, :T1, i),
            gva(i + 1, 1),
            gvb(i + 1, 1),
            (gh(i + 1, 1), gh(i, 1))...,
        ) for i in 1:Nx
    ]

    e_T2s = [
        Expr(
            :ref,
            Expr(:ref, :T2, j),
            gha(Nx + 1, j + 1),
            ghb(Nx + 1, j + 1),
            (gv(Nx + 2, j), gv(Nx + 2, j + 1))...,
        ) for j in 1:Ny
    ]
    e_T3s = [
        Expr(
            :ref,
            Expr(:ref, :T3, i),
            gva(i + 1, Ny + 1),
            gvb(i + 1, Ny + 1),
            (gh(i, Ny + 2), gh(i + 1, Ny + 2))...,
        ) for i in 1:Nx
    ]
    e_T4s = [
        Expr(
            :ref,
            Expr(:ref, :T4, j),
            gha(1, j + 1),
            ghb(1, j + 1),
            (gv(1, j + 1), gv(1, j))...,
        ) for j in 1:Ny
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i, j),
            phys(i, j)...,
            gha(i + 1, j + 1),
            gva(i + 1, j + 1),
            gha(i, j + 1),
            gva(i + 1, j),
        ) for i in 1:Nx, j in 1:Ny
    ]

    e_MSp = [
        Expr(
            :ref,
            Expr(:ref, :MSP, i, j),
            ghb(i + 1, j + 1),
            gvb(i + 1, j + 1),
            ghb(i, j + 1),
            gvb(i + 1, j),
            phys(i, j)...,
        ) for i in 1:Nx, j in 1:Ny
    ]

    e_C1 = Expr(:ref, :C1, gh(1, 1), gv(1, 1))
    e_C2 = Expr(:ref, :C2, gh(Nx + 1, 1), gv(Nx + 2, 1))
    e_C3 = Expr(:ref, :C3, gh(Nx + 1, Ny + 2), gv(Nx + 2, Ny + 1))
    e_C4 = Expr(:ref, :C4, gh(1, Ny + 2), gv(1, Ny + 1))

    e_einsum = Expr(
        :call,
        :*,
        e_C1,
        e_C2,
        e_C3,
        e_C4,
        e_T1s...,
        e_T2s...,
        e_T3s...,
        e_T4s...,
        e_MS...,
        e_MSp...,
    )

    @info e_einsum
    quote
        @tensoropt rv = $e_einsum
    end
end
@generated function _contractall(
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:AbsTen{0}},
) where {N,T1,T2}
    symb = (s, i) -> Symbol("$(s)_$i")

    e_ARU = [
        Expr(:ref, Expr(:ref, :ARU, i), symb(:n, i + 1), symb(:u, i + 1), symb(:u, i)) for
        i in 1:N
    ]
    e_ARD = [
        Expr(:ref, Expr(:ref, :ARD, i), symb(:d, i + 1), symb(:d, i), symb(:s, i + 1)) for
        i in 1:N
    ]

    e_MS = [
        Expr(
            :ref, Expr(:ref, :MS, i), symb(:h, i), symb(:s, i), symb(:h, i - 1), symb(:n, i)
        ) for i in 1:(N + 1)
    ]

    e_ACU = Expr(:ref, :ACU, symb(:n, 1), symb(:u, 1), symb(:u, 0))
    e_ACD = Expr(:ref, :ACD, symb(:d, 1), symb(:d, 0), symb(:s, 1))
    e_FL = Expr(:ref, :FL, symb(:h, 0), symb(:u, 0), symb(:d, 0))
    e_FR = Expr(:ref, :FR, symb(:h, N + 1), symb(:u, N + 1), symb(:d, N + 1))

    e_einsum = Expr(:call, :*, e_FL, e_FR, e_ACU, e_ACD, e_MS..., e_ARU..., e_ARD...)

    quote
        @tensoropt rv = $e_einsum
    end
end

function _contractall(
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:CompositeTensor{2}},
) where {N,T1,T2}
    MSu = first.(MS)
    MSd = last.(MS)

    return _contractall(FL, FR, ACU, ARU, ACD, ARD, MSu, MSd)
end

@generated function _contractall(
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:AbsTen{NInd}},
    MSP::AbstractArray{<:TenAbs{NInd}},
) where {N,NInd,T1,T2}
    symb = (s, i) -> Symbol("$(s)_$i")

    e_ARU = [
        Expr(
            :ref,
            Expr(:ref, :ARU, i),
            symb(:n, i + 1), # north bond of bulk
            symb(:np, i + 1), # north bond of adjoint bulk
            symb(:u, i + 1),
            symb(:u, i),
        ) for i in 1:N
    ]
    e_ARD = [
        Expr(
            :ref,
            Expr(:ref, :ARD, i),
            symb(:d, i + 1),
            symb(:d, i),
            symb(:s, i + 1), # south bond of bulk
            symb(:sp, i + 1), # south bond of adjoint bulk
        ) for i in 1:N
    ]

    phys = i -> begin
        if NInd == 1
            [symb(:p, i)]
        elseif NInd == 2
            [symb(:p1, i), symb(:p2, i)]
        end
    end

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i),
            phys(i)...,
            symb(:h, i),
            symb(:s, i),
            symb(:h, i - 1),
            symb(:n, i),
        ) for i in 1:(N + 1)
    ]
    e_MSp = [
        Expr(
            :ref,
            Expr(:ref, :MSP, i),
            symb(:hp, i),
            symb(:sp, i),
            symb(:hp, i - 1),
            symb(:np, i),
            phys(i)...,
        ) for i in 1:(N + 1)
    ]

    e_ACU = Expr(:ref, :ACU, symb(:n, 1), symb(:np, 1), symb(:u, 1), symb(:u, 0))
    e_ACD = Expr(:ref, :ACD, symb(:d, 1), symb(:d, 0), symb(:s, 1), symb(:sp, 1))
    e_FL = Expr(:ref, :FL, symb(:h, 0), symb(:hp, 0), symb(:u, 0), symb(:d, 0))
    e_FR = Expr(
        :ref, :FR, symb(:h, N + 1), symb(:hp, N + 1), symb(:u, N + 1), symb(:d, N + 1)
    )

    e_einsum = Expr(
        :call, :*, e_FL, e_FR, e_ACU, e_ACD, e_MS..., e_MSp..., e_ARU..., e_ARD...
    )
    @info e_einsum

    quote
        @tensoropt rv = $e_einsum
    end
end
