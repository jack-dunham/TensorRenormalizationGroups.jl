function KrylovKit.initialize(network, algorithm::AbstractCornerMethod; kwargs...)
    if size(network) > (1, 1) && isa(algorithm, FPCM)
        println("The FPCM for non-trivial unit cells is experimental 
                    and may not give correct resultsworking")
    end

    primary_tensors = inittensors(network, algorithm; kwargs...)
    permuted_tensors = initpermuted(primary_tensors)
    svals = initerror(primary_tensors)
    return CornerMethodRuntime(primary_tensors, permuted_tensors, svals)
end

reset!(runtime::CornerMethodRuntime, network) = updatenetwork!(runtime, network)

function updatenetwork!(runtime::CornerMethodRuntime, network)
    updatenetwork!(runtime.primary, network)
    updatenetwork!(runtime.permuted, permutedims(swapaxes(network)))
    map(runtime.svals) do sval
        broadcast(sval) do s
            copy!(s, isometry(codomain(s), domain(s)))
            # one!(s)
        end
    end
    return runtime
end

function updatenetwork!(tensors::CornerMethodTensors, network)
    foreach(tensors.network, network) do t1, t2
        copy!(t1, t2)
    end
    return tensors
end

function inittensors(network, alg::AbstractCornerMethod; randinit=alg.randinit)
    # Convert the bond dimension into an IndexSpace
    chi = dimtospace(spacetype(network), alg.bonddim)

    corners = initcorners(network, chi; randinit=randinit)
    edges = initedges(network, chi; randinit=randinit)
    projectors = initprojectors(network, chi)

    return CornerMethodTensors(corners, edges, projectors, network)
end

function initpermuted(tensors::CornerMethodTensors)
    chi = chispace(tensors)

    # First swap the horizontal and vertical axes:
    transposed_corners = map(permutedims, tensors.corners)
    transposed_edges = map(permutedims, tensors.edges)

    # Then permute the indices of each tensor on the lattice:
    cs = map(transposed_corners) do c
        return broadcast(c) do t
            return permutedom(t, (2, 1))
        end
    end

    # The order of the tensors needs adjusted such that they appear in the correct place
    corners = Corners((cs[1], cs[4], cs[3], cs[2]))
    edges = Edges(reverse(transposed_edges))

    # Need to swap the axes bonds and transpose the network
    network = permutedims(swapaxes(tensors.network))

    # Projectors are easiest to construct assuming a transposed and permuted unit cell 
    projectors = initprojectors(network, chi)

    return CornerMethodTensors(corners, edges, projectors, network)
end

## CORNERS 

function initcorners(network, chi::S; randinit::Bool=false) where {S<:IndexSpace}
    corner_tensors = map(1:4) do i
        chis = (chi, chi', chi, chi')

        corners = broadcast(network) do site
            T = scalartype(network)
            cout = TensorMap{T}(undef, one(S), chis[i] * chis[i])

            if randinit
                randn!(cout)
            else
                tenp = rotate(site, -i + 1)
                init_single_corner!(cout, tenp)
            end
            normalize!(cout)
            return cout
        end

        return corners
    end

    corners = Corners(corner_tensors...)

    return randomize_if_zero!(corners)
end
function init_single_corner!(cout, ten::AbstractTensorMap{T,S}) where {T,S}
    d = virtualspace(ten)

    u1 = get_embedding_isometry(d[1], domain(cout)[1])
    u2 = get_embedding_isometry(d[2], domain(cout)[2])

    u3 = get_removal_isometry(d[3])
    u4 = get_removal_isometry(d[4])

    corner = init_single_corner!(cout, ten, u1, u2, u3, u4)

    return corner
end

function init_single_corner!(t_dst, t_src::CompositeTensor{2}, ue, us, uw, un)
    top, bot = t_src
    return __init_single_corner!(t_dst, top, bot, ue, us, uw, un)
end
function init_single_corner!(t_dst, t_src::TensorMap, ue, us, uw, un)
    return __init_single_corner!(t_dst, t_src, ue, us, uw, un)
end

function __init_single_corner!(
    t_dst, t_src::TensorMap{<:Number,<:IndexSpace,0,4}, ue, us, uw, un
)
    @tensoropt t_dst[o1 o2] = t_src[e s w n] * ue[e; o1] * us[s; o2] * uw[w] * un[n]
    return t_dst
end

function __init_single_corner!(
    t_dst, t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{1,4},T2<:AbsTen{4,1}}
    @tensoropt t_dst[o1 o2] =
        t1[k; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k] *
        ue[e1 e2; o1] *
        us[s1 s2; o2] *
        uw[w1 w2] *
        un[n1 n2]
    return t_dst
end

function __init_single_corner!(
    t_dst, t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{2,4},T2<:AbsTen{4,2}}
    @tensoropt t_dst[o1 o2] =
        t1[k b; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k b] *
        ue[e1 e2; o1] *
        us[s1 s2; o2] *
        uw[w1 w2] *
        un[n1 n2]
    return t_dst
end

##

## EDGES

function initedges(network, chi::IndexSpace; randinit::Bool=false)
    edge_tensors = map(1:4) do i
        edges = broadcast(network) do site
            tenp = rotate(site, -i + 1)

            # Need north bond of tensor below (south bond)
            d = swap(virtualspace(tenp)[2])

            T = scalartype(site)

            eout = TensorMap{T}(undef, d, chi * chi')

            if randinit
                randn!(eout)
            else
                eout = init_single_edge!(eout, tenp)
            end

            normalize!(eout)

            return eout
        end
        return edges
    end

    edges = Edges(edge_tensors...)

    return randomize_if_zero!(edges)
end

function init_single_edge!(eout, tenp)
    d = virtualspace(tenp)

    u1 = get_embedding_isometry(d[1], domain(eout)[1])
    # u1 = id(d[1])
    u2 = isometry(swap(d[2]), swap(d[2]))
    u3 = get_embedding_isometry(d[3], domain(eout)[2])
    # u3 = id(d[3])
    u4 = get_removal_isometry(d[4])

    edge = init_single_edge!(eout, tenp, u1, u2, u3, u4)

    return edge
end

function init_single_edge!(t_dst, t_src::CompositeTensor{2}, ue, us, uw, un)
    top, bot = t_src
    return __init_single_edge!(t_dst, top, bot, ue, us, uw, un)
end
function init_single_edge!(t_dst, t_src::TensorMap, ue, us, uw, un)
    return __init_single_edge!(t_dst, t_src, ue, us, uw, un)
end

function __init_single_edge!(tdst, tsrc::AbsTen{0,4}, ue, us, uw, un)
    @tensoropt tdst[s; o1 o2] := tsrc[e s w n] * ue[e; o1] * uw[w; o2] * un[n]
    # @tensoropt t_dst[ss; o1 o2] = t_src[e s w n] * ue[e; o1] * us[ss; s] * uw[w; o2] * un[n]
    return tdst
end
function __init_single_edge!(
    t_dst, t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{1,4},T2<:AbsTen{4,1}}
    @tensoropt t_dst[ss1 ss2; o1 o2] =
        t1[k; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
end
function __init_single_edge!(
    t_dst, t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{2,4},T2<:AbsTen{4,2}}
    @tensoropt t_dst[ss1 ss2; o1 o2] =
        t1[k b; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k b] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
end

##

## PROJECTORS

function initprojectors(network, chi::IndexSpace)
    _, south_bonds, _, north_bonds = virtualspace(network)

    T = scalartype(network)

    UL = construct_projector(T, adjoint(chi), north_bonds, (-1, -1))
    VL = construct_projector(T, chi, south_bonds, (-1, 0))
    UR = construct_projector(T, chi, north_bonds, (1, -1))
    VR = construct_projector(T, adjoint(chi), south_bonds, (1, 0))

    return Projectors(UL, VL, UR, VR)
end

function construct_projector(T, chi, sitebonds, incr)
    rv = broadcast(circshift(sitebonds, incr)) do bonds
        return TensorMap{T}(undef, chi * bonds, chi)
    end
    return rv
end

##

## ERROR

initerror(ctm::CornerMethodTensors) = initerror(ctm.corners)
function initerror(corners::Corners)
    # Permute into some form compatible with tsvd
    svals = map(corners) do corn
        broadcast(corn) do site
            s = permute(site, ((1,), (2,)))
            return TensorMap(rand, scalartype(s), codomain(s), domain(s))
            # _, rv, _ = tsvd(permute(site, ((1,), (2,))))
            # return rv
        end
    end

    return CornerSingularValues(svals)
end

##

## UTILS

randomize_if_zero!(corners::Corners) = randomize_if_zero!(corners, :C)
randomize_if_zero!(edges::Edges) = randomize_if_zero!(edges, :T)

function randomize_if_zero!(corners_or_edges, type::Symbol)
    for (i, c_or_e) in enumerate(corners_or_edges)
        map(x -> randomize_if_zero!(c_or_e, type, i, x), CartesianIndices(c_or_e))
    end

    return corners_or_edges
end
function randomize_if_zero!(uc, type, i, ind)
    ten = uc[ind]
    if ten ≈ zero(ten) || isnan(norm(ten)) || isinf(norm(ten))
        @info "Ill-conditioned tensor $type$i at site $(Tuple(ind)); using a random tensor instead."
        randn!(ten)
    end
    return uc
end

##
