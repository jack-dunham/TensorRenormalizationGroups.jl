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

function initcorners(
    network::UnitCell{G}, chi::S; randinit::Bool=false
) where {G,S<:IndexSpace}
    TType = tensormaptype(S, 0, 2, scalartype(network))

    UType = UnitCell{G,TType,Matrix{TType}}

    chis = (chi, chi', chi, chi')
    corner_tensors = map((1, 2, 3, 4)) do i
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

    corners = Corners(corner_tensors)

    return randomize_if_zero!(corners)::Corners{UType}
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

function edgetype(network::AbstractUnitCell{G,<:CompositeTensor{2}}, chi) where {G}
    return tensormaptype(typeof(chi), 2, 2, scalartype(network))
end
function edgetype(network::AbstractUnitCell{G,<:TensorMap}, chi) where {G}
    return tensormaptype(typeof(chi), 1, 2, scalartype(network))
end

function initedges(network::UnitCell{G}, chi::IndexSpace; randinit::Bool=false) where {G}
    TType = edgetype(network, chi)

    UType = UnitCell{G,TType,Matrix{TType}}

    edge_tensors = map((1, 2, 3, 4)) do i
        edges = similar(network, TType)

        broadcast!(edges, network) do site
            tenp = rotate(site, -i + 1)

            eout = init_single_edge(chi * chi', tenp)

            if randinit
                randn!(eout)
            end

            normalize!(eout)

            return eout
        end
        return edges
    end

    edges = Edges(edge_tensors)

    return randomize_if_zero!(edges)::Edges{UType}
end

function init_single_edge(dstdom, tenp)
    d = virtualspace(tenp)

    u1 = get_embedding_isometry(d[1], dstdom[1])
    # u1 = id(d[1])
    u2 = isometry(swap(d[2]), swap(d[2]))
    u3 = get_embedding_isometry(d[3], dstdom[2])
    # u3 = id(d[3])
    u4 = get_removal_isometry(d[4])

    edge = init_single_edge(tenp, u1, u2, u3, u4)

    return edge
end

function init_single_edge(tsrc::CompositeTensor{2}, ue, us, uw, un)
    top, bot = tsrc
    return __init_single_edge(top, bot, ue, us, uw, un)
end
function init_single_edge(tsrc::AbsTen{0,4}, ue, us, uw, un)
    @tensoropt tdst[s; o1 o2] := tsrc[e s w n] * ue[e; o1] * uw[w; o2] * un[n]
    # @tensoropt t_dst[ss; o1 o2] = t_src[e s w n] * ue[e; o1] * us[ss; s] * uw[w; o2] * un[n]
    return tdst
end
function __init_single_edge(
    t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{1,4},T2<:AbsTen{4,1}}
    @tensoropt tdst[ss1 ss2; o1 o2] :=
        t1[k; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
    return tdst
end
function __init_single_edge(
    t1::T1, t2::T2, ue, us, uw, un
) where {T1<:AbsTen{2,4},T2<:AbsTen{4,2}}
    @tensoropt tdst[ss1 ss2; o1 o2] :=
        t1[k b; e1 s1 w1 n1] *
        t2[e2 s2 w2 n2; k b] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
    return tdst
end

##

## PROJECTORS
function projtype(network::AbstractUnitCell{G,<:CompositeTensor{2}}, chi) where {G}
    return tensormaptype(typeof(chi), 3, 1, scalartype(network))
end
function projtype(network::AbstractUnitCell{G,<:TensorMap}, chi) where {G}
    return tensormaptype(typeof(chi), 2, 1, scalartype(network))
end

function initprojectors(network, chi::IndexSpace)
    _, south_bonds, _, north_bonds = virtualspace(network)

    UL = similar(network, projtype(network, chi))
    VL = similar(network, projtype(network, chi))
    UR = similar(network, projtype(network, chi))
    VR = similar(network, projtype(network, chi))

    construct_projector!(UL, adjoint(chi), north_bonds, (-1, -1))
    construct_projector!(VL, chi, south_bonds, (-1, 0))
    construct_projector!(UR, chi, north_bonds, (1, -1))
    construct_projector!(VR, adjoint(chi), south_bonds, (1, 0))

    return Projectors(UL, VL, UR, VR)
end

function construct_projector!(ucdst, chi, sitebonds, incr)
    T = scalartype(ucdst)

    broadcast!(ucdst, circshift(sitebonds, incr)) do bonds
        return TensorMap{T}(undef, chi * bonds, chi)
    end
    return ucdst
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

randomize_if_zero!(corners::C) where {C<:Corners} = randomize_if_zero!(corners, :C)::C
function randomize_if_zero!(edges::E) where {E<:Edges}
    randomize_if_zero!(edges, :T)
    return edges
end

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
