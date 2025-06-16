# DONE (NEEDS RESTRUCTURING)
abstract type AbstractFixedPoints end

# Fixed points of a transfer matrix
struct FixedPoints{A<:AbUnCe{<:TenAbs{2}}} <: AbstractFixedPoints
    left::A
    right::A
    function FixedPoints(left::A, right::A) where {A}
        check_size_allequal(left, right)
        return new{A}(left, right)
    end
end

function FixedPoints(f, mps::MPS, network::AbstractNetwork)
    mps_tensor = getcentral(mps)

    left = initleft.(f, mps_tensor, network)
    right = initright.(f, mps_tensor, network)

    return FixedPoints(left, right)
end

Base.similar(fps::FixedPoints) = FixedPoints(similar(fps.left), similar(fps.right))

TensorKit.scalartype(::Type{<:FixedPoints{A}}) where {A} = scalartype(A)

initleft(f, mps, bulk) = initfixedpoint(f, mps, bulk, :left)
initright(f, mps, bulk) = initfixedpoint(f, mps, bulk, :right)

function initfixedpoint(f, mps, bulk, leftright::Symbol)
    if leftright === :left
        bulkind = 3
        mpsind = 2
    elseif leftright === :right
        bulkind = mpsind = 1
    else
        throw(ArgumentError(""))
    end

    cod = virtualspace(bulk, bulkind)
    dom = domain(mps)[mpsind]' * domain(mps)[mpsind]

    return f(promote_type(scalartype(mps), scalartype(bulk)), cod, dom)
end

function renorm(cb, ca, fl, fr)
    c_out = hcapply!(similar(ca), ca, fl, fr)
    N = dot(cb, c_out)
    return N
end

function hcapply!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{1,2}, fr::AbsTen{1,2})
    @tensoropt hc[dr dl] = c[ur ul] * fl[m; ul dl] * fr[m; ur dr]
    return hc
end
function hcapply!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{2,2}, fr::AbsTen{2,2})
    @tensoropt hc[dr dl] = c[ur ul] * fl[m1 m2; ul dl] * fr[m1 m2; ur dr]
    return hc
end

function fixed_point_norm(cd::C, cu::C, fl::F, fr::F) where {C<:AbsTen{0,2},F<:AbsTen{1,2}}
    @tensoropt n = cu[ur ul] * fl[m; ul dl] * fr[m; ur dr] * conj(cd[dr dl])
    return n
end

function fixed_point_norm(cd::C, cu::C, fl::F, fr::F) where {C<:AbsTen{0,2},F<:AbsTen{2,2}}
    @tensoropt n = cu[ur ul] * fl[m1 m2; ul dl] * fr[m1 m2; ur dr] * conj(cd[dr dl])
    return n
end
# function normalize(cd, cu, fl, fr)
#     @tensoropt hc[dr dl] = cu[ur ul] * fl[m1 m2; ul dl] * fr[m1 m2; ur dr] * conj(cd[dr dl])
# end

function fixedpoints(mps::MPS, network, f0=FixedPoints(rand, mps, network))
    return fixedpoints!(f0, mps, network)
end

#this is now the correct env func

function fixedpoints!(
    fpoints::FixedPoints,
    mps::MPS,
    network;
    ishermitian=forcehermitian(fpoints, mps, network),
)
    AL, C, AR, _ = unpack(mps)

    # TransferMatrix(AL[1, 1], network[1, 1], AL[1, 1])

    tm_left = TransferMatrix.(AL, network, circshift(AL, (0, -1)))
    tm_right = TransferMatrix.(AR, network, circshift(AR, (0, -1)))

    return fixedpoints!(fpoints, tm_left, tm_right, C; ishermitian=ishermitian)
end

function fixedpoints!(
    fpoints::FixedPoints,
    tm_left::AbstractTransferMatrices,
    tm_right::AbstractTransferMatrices,
    C::AbstractUnitCell;
    ishermitian=forcehermitian(fpoints, tm_left, tm_right, C),
)
    FL = fpoints.left
    FR = fpoints.right

    Nx, Ny = size(C)

    for y in axes(eachindex(C), 2)
        left, Ls, linfo = eigsolve(
            z -> leftsolve(z, tm_left[:, y]), FL[1, y], 1, :LM; eager=true
        )
        right, Rs, rinfo = eigsolve(
            z -> rightsolve(z, tm_right[:, y]), FR[Nx, y], 1, :LM; eager=true
        )

        FL[1, y] = Ls[1]
        FR[end, y] = Rs[1]

        @debug "Fixed point leading eigenvalues:" left = left[1] right = right[1]
        @debug "Fixed point convergence info:" left = linfo right = rinfo

        for x in 2:Nx
            multransfer!(FL[x, y], FL[x - 1, y], tm_left[x - 1, y])
        end

        NN = fixed_point_norm(C[Nx, y + 1], C[Nx, y], Ls[1], Rs[1]) # Should be positive?

        @debug "" norm = NN

        # TODO: use normalize_fixed_points
        if isa(NN, AbstractFloat) && NN < 0.0
            NN = sqrt(sign(NN) * NN)
            # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
            rmul!(FL[1, y], 1 / NN) #correct NN
            rmul!(FR[end, y], 1 / -NN) #correct NN
        else
            NN = sqrt(NN)
            # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
            rmul!(FL[1, y], 1 / NN) #correct NN
            rmul!(FR[end, y], 1 / NN) #correct NN
        end

        # # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
        # rmul!(FL[1, y], 1 / NN) #correct NN
        # rmul!(FR[end, y], 1 / NN) #correct NN

        for x in (Nx - 1):-1:1
            multransfer!(FR[x, y], tm_right[x + 1, y], FR[x + 1, y])

            NN = normalize_fixed_points!(FL[x + 1, y], FR[x, y], C[x, y + 1], C[x, y])

            # NN = renorm(C[x, y + 1], C[x, y], FL[x + 1, y], FR[x, y])
            #
            # rmul!(FL[x + 1, y], 1 / sqrt(NN))
            # rmul!(FR[x, y], 1 / sqrt(NN))
        end

        # First x seems to be normalized, but not second x
        for x in 1:Nx
            NN = fixed_point_norm(C[x, y + 1], C[x, y], FL[x + 1, y], FR[x, y])
            # println(NN)
        end
    end
    return fpoints
end

function normalize_fixed_points!(FL, FR, C1, C2)
    # NN = renorm(C1, C2, FL, FR)
    NN = fixed_point_norm(C1, C2, FL, FR)
    if isa(NN, AbstractFloat)
        s = sign(NN)
        N1 = sqrt(s * NN)
        N2 = s * sqrt(s * NN)

    else
        N1 = sqrt(NN)
        N2 = sqrt(NN)
    end
    rmul!(FL, 1 / N1) #correct NN
    rmul!(FR, 1 / N2) #correct NN
    return FL, FR
end
# FL[x] * T[x] = FL[x + 1]
function leftsolve(f0, Ts)
    fnew = f0

    for T in Ts
        cod, dom = rightspace(T)
        fold = fnew
        fnew = similar(fnew, cod, dom)
        multransfer!(fnew, fold, T)
    end

    return fnew
end
function rightsolve(f0, Ts)
    fnew = f0

    for T in reverse(Ts)
        cod, dom = leftspace(T)
        fold = fnew
        fnew = similar(fnew, cod, dom)
        multransfer!(fnew, T, fold)
    end

    return fnew
end

# This needs updated
# function fptest(FL, FR, A, M)
#     nx, ny = size(M)
#     AL, C, AR, _ = unpack(A)
#     for x in 1:nx, y in 1:ny
#         println(
#             "Left: ",
#             isapprox(
#                 normalize(FL[x, y]), normalize(fpsolve(FL[x, y], AL, M, x, y)); atol=1e-3
#             ),
#         )
#         println(
#             "Right: ",
#             isapprox(normalize(FR[x, y]), normalize(fpsolve(FR[x, y], AR, M, x, y))),
#         )
#     end
# end
