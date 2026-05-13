struct TransferMatrix{A<:TenAbs{2},M}
    above::A
    middle::M
    below::A
end

function Base.eltype(t::TransferMatrix)
    return promote_type(eltype(t.above), eltype(t.middle), eltype(t.below))
end

const AbstractTransferMatrices{T<:TransferMatrix} = AbUnCe{T}

function rightspace(tm::TransferMatrix)
    dom = domain(tm.above, 1) * domain(tm.below, 1)'
    cod = mapadjoint(virtualspace(tm.middle, 1))
    return cod, dom
end

# This is the space that the new 
function leftspace(tm::TransferMatrix)
    dom = domain(tm.above, 2) * domain(tm.below, 2)'
    cod = mapadjoint(virtualspace(tm.middle, 3))
    return cod, dom
end

## MULTIPLICATION
function Base.:*(f, transfer::TransferMatrix)
    cod, dom = rightspace(transfer)
    fdst = similar(f, cod, dom)
    return multransfer!(fdst, f, transfer)
end
function Base.:*(transfer::TransferMatrix, f)
    cod, dom = leftspace(transfer)
    fdst = similar(f, cod, dom)
    return multransfer!(fdst, transfer, f)
end

function multransfer!(fdst, t1, t2)
    left, right, mps_top, mps_bot = _multransfer_tensors(t1, t2)
    return multransfer!(fdst, left, right, mps_top, mps_bot)
end

function _multransfer_tensors(left_point::AbsTen, transfer::TransferMatrix)
    mps_top = transfer.above
    network_tensor = transfer.middle
    mps_bot = transfer.below

    return left_point, network_tensor, mps_top, mps_bot
end
function _multransfer_tensors(transfer::TransferMatrix, right_point::AbsTen)
    mps_top = transfer.above
    network_tensor = transfer.middle
    mps_bot = transfer.below

    return network_tensor, right_point, mps_top, mps_bot
end

## LEFT (FL' = FL * M)

function multransfer!(
    fdst::AbsTen{1,2}, fsrc::AbsTen{1,2}, m::AbsTen{0,4}, at::AbsTen{1,2}, ab::AbsTen{1,2}
)
    @tensoropt fdst[D1; x2 x4] =
        fsrc[D3; x1 x3] * at[D4; x2 x1] * m[D1 D2 D3 D4] * (ab')[x4 x3; D2]
    return fdst
end

function multransfer!(fdst, fsrc, m::CompositeTensor{2}, at, ab)
    top, bot = m
    return multransfer!(fdst, fsrc, top, bot, at, ab)
end

function multransfer!(
    fdst::AbsTen{2,2},
    fsrc::AbsTen{2,2},
    ma::AbsTen{1,4},
    mb::AbsTen{4,1},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D1 E1; x2 x4] =
        fsrc[D3 E3; x1 x3] *
        at[D4 E4; x2 x1] *
        ma[k; D1 D2 D3 D4] *
        mb[E1 E2 E3 E4; k] *
        (ab')[x4 x3; D2 E2]
    return fdst
end
function multransfer!(
    fdst::AbsTen{2,2},
    fsrc::AbsTen{2,2},
    ma::AbsTen{2,4},
    mb::AbsTen{4,2},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D1 E1; x2 x4] =
        fsrc[D3 E3; x1 x3] *
        at[D4 E4; x2 x1] *
        ma[k b; D1 D2 D3 D4] *
        mb[E1 E2 E3 E4; k b] *
        (ab')[x4 x3; D2 E2]
    return fdst
end

# M * FR
function multransfer!(
    fdst::AbsTen{1,2}, m::AbsTen{0,4}, fsrc::AbsTen{1,2}, at::AbsTen{1,2}, ab::AbsTen{1,2}
)
    @tensoropt fdst[D3; x1 x3] =
        fsrc[D1; x2 x4] * at[D4; x2 x1] * m[D1 D2 D3 D4] * (ab')[x4 x3; D2]
    return fdst
end

## DOUBLE LAYER
function multransfer!(fdst, m::CompositeTensor{2}, fsrc, at, ab)
    top, bot = m
    return multransfer!(fdst, top, bot, fsrc, at, ab)
end

function multransfer!(
    fdst::AbsTen{2,2},
    ma::AbsTen{1,4},
    mb::AbsTen{4,1},
    fsrc::AbsTen{2,2},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D3 E3; x1 x3] =
        fsrc[D1 E1; x2 x4] *
        at[D4 E4; x2 x1] *
        ma[k; D1 D2 D3 D4] *
        mb[E1 E2 E3 E4; k] *
        (ab')[x4 x3; D2 E2]
    return fdst
end
function multransfer!(
    fdst::AbsTen{2,2},
    ma::AbsTen{2,4},
    mb::AbsTen{4,2},
    fsrc::AbsTen{2,2},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D3 E3; x1 x3] =
        fsrc[D1 E1; x2 x4] *
        at[D4 E4; x2 x1] *
        ma[k b; D1 D2 D3 D4] *
        mb[E1 E2 E3 E4; k b] *
        (ab')[x4 x3; D2 E2]
    return fdst
end
