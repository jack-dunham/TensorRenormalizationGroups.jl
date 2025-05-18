symmetrise_corner!(x) = axpby!(1//2, cj(permutedom(x, (2, 1))), 1//2, x)
function symmetrise_edge!(x)
    return axpby!(1//2, isometry(codomain(x), codomain(x)') * transpose(x)', 1//2, x)
end

function projectcorner(c, t, p)
    dstdom = domain(t, 1) * domain(p, 1)
    cdst = similar(c, codomain(c), dstdom)
    return projectcorner!(cdst, c, t, p)
end

#=
     1   3  2
    C --- T ---
    |2    |
    *--V--*
       |
=#
function projectcorner!(c_dst, c_src, t, uv)
    if c_dst === c_src
        c_copy = deepcopy(c_src)
        _projectcorner!(c_dst, c_copy, t, uv)
    else
        _projectcorner!(c_dst, c_src, t, uv)
    end
    # nor = norm(cj(permutedom(c_dst, (2, 1))) - c_dst)
    # symmetrise_corner!(c_dst)
    return c_dst
end

function _projectcorner!(c_dst, c_src, t::AbsTen{1,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2; h0 h1] * uv[v1 v2; v0]
    return c_dst
end
function _projectcorner!(c_dst, c_src, t::AbsTen{2,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2 x; h0 h1] * uv[v1 v2 x; v0]
    return c_dst
end

function projectedge(t, m, u, v)
    coddst = reverse(ProductSpace(virtualspace(m, 1))')
    domdst = domain(v, 1) * domain(u, 1)
    tdst = similar(t, coddst, domdst)
    return projectedge!(tdst, t, m, u, v)
end
function projectedge!(t_dst, t_src, m, u, v)
    if t_dst === t_src
        t_copy = deepcopy(t_src)
        _projectedge!(t_dst, t_copy, m, u, v)
    else
        _projectedge!(t_dst, t_src, m, u, v)
    end
    # symmetrise_edge!(t_dst)
    return t_dst
end

## SINGLE LAYER

function _projectedge!(
    t_dst::T, t_src::T, m::M, u::UV, v::UV
) where {T<:AbsTen{1,2},M<:AbsTen{0,4},UV<:AbsTen{2,1}}
    @tensoropt t_dst[h0; v0_d v0_u] =
        t_src[h1; v3 v1] * m[h0 v4 h1 v2] * u[v1 v2; v0_u] * v[v3 v4; v0_d]
    return t_dst
end

## DOUBLE LAYER

function _projectedge!(
    t_dst::T, t_src::T, m::M, u::UV, v::UV
) where {T<:AbsTen{2,2},M<:CompositeTensor{2},UV<:AbsTen{3,1}}
    top, bot = m
    return _projectedge!(t_dst, t_src, top, bot, u, v)
end

function _projectedge!(
    t_dst::T, t_src::T, ma::MA, mb::MB, u::UV, v::UV
) where {T<:AbsTen{2,2},MA<:AbsTen{1,4},MB<:AbsTen{4,1},UV<:AbsTen{3,1}}
    @tensoropt (
        k => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) t_dst[D1 D5; x4 x2] =
        t_src[D3 D7; x3 x1] *
        ma[k; D1 D2 D3 D4] *
        mb[D5 D6 D7 D8; k] *
        u[x1 D4 D8; x2] *
        v[x3 D2 D6; x4]

    return t_dst
end

function _projectedge!(
    t_dst::T, t_src::T, ma::MA, mb::MB, u::UV, v::UV
) where {T<:AbsTen{2,2},MA<:AbsTen{2,4},MB<:AbsTen{4,2},UV<:AbsTen{3,1}}
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) t_dst[D1 D5; x4 x2] =
        t_src[D3 D7; x3 x1] *
        ma[k b; D1 D2 D3 D4] *
        mb[D5 D6 D7 D8; k b] *
        u[x1 D4 D8; x2] *
        v[x3 D2 D6; x4]

    return t_dst
end

## SINGLE LAYER METHODS

# O(χ^2 D^8) or D^10
function halfcontract(
    C1_00::C, T1_10::T, T1_20::T, C2_30::C, T4_01::T, M_11::M, M_21::M, T2_31::T
) where {C<:AbsTen{0,2},T<:AbsTen{1,2},M<:AbsTen{0,4}}
    @tensoropt out[v8 v7; v5 v6] :=
        C1_00[h1 v1] *
        T1_10[v2; h2 h1] *
        T1_20[v3; h3 h2] *
        C2_30[h3 v4] *
        T4_01[h4; v5 v1] *
        M_11[h5 v6 h4 v2] *
        M_21[h6 v7 h5 v3] *
        T2_31[h6; v4 v8]

    return out
end

## DOUBLE LAYER METHODS

function halfcontract(
    C1_00::C, T1_10::T, T1_20::T, C2_30::C, T4_01::T, M_11::M, M_21::M, T2_31::T
) where {C<:AbsTen{0,2},T<:AbsTen{2,2},M<:CompositeTensor{2}}
    M_11a, M_11b = M_11
    M_21a, M_21b = M_21
    return halfcontract(
        C1_00, T1_10, T1_20, C2_30, T4_01, M_11a, M_11b, M_21a, M_21b, T2_31
    )
end

function halfcontract(
    C1_00::C,
    T1_10::T,
    T1_20::T,
    C2_30::C,
    T4_01::T,
    M_11a::MA,
    M_11b::MB,
    M_21a::MA,
    M_21b::MB,
    T2_31::T,
) where {C<:AbsTen{0,2},T<:AbsTen{2,2},MA<:AbsTen{1,4},MB<:AbsTen{4,1}}
    @tensoropt (
        k1 => 2,
        k2 => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        x7 => D,
    ) top[x7 D6 E6; x6 D2 E2] :=
        M_11a[k1; D1 D2 D3 D4] *
        M_21a[k2; D5 D6 D1 D8] *
        M_11b[E1 E2 E3 E4; k1] *
        M_21b[E5 E6 E1 E8; k2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]

    return top
end

function halfcontract(
    C1_00::C,
    T1_10::T,
    T1_20::T,
    C2_30::C,
    T4_01::T,
    M_11a::MA,
    M_11b::MB,
    M_21a::MA,
    M_21b::MB,
    T2_31::T,
) where {C<:AbsTen{0,2},T<:AbsTen{2,2},MA<:AbsTen{2,4},MB<:AbsTen{4,2}}
    @tensoropt (
        k1 => 2,
        k2 => 2,
        b1 => 2,
        b2 => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        x7 => D,
    ) top[x7 D6 E6; x6 D2 E2] :=
        M_11a[k1 b1; D1 D2 D3 D4] *
        M_21a[k2 b2; D5 D6 D1 D8] *
        M_11b[E1 E2 E3 E4; k1 b1] *
        M_21b[E5 E6 E1 E8; k2 b2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]

    return top
end

projectcorner2d(c, t1, t4, m, uh, uv) = projectcorner2d!(similar(c), c, t1, t4, m, uh, uv)

function projectcorner2d!(c_dst, c_src, t1, t4, m, uh, uv)
    @tensoropt c_dst[h0 v0] =
        c_src[h1 v1] *
        t1[v2; h2 h1] *
        uh[h2 h4; h0] *
        t4[h3; v3 v1] *
        m[h4 v4 h3 v2] *
        uv[v3 v4; v0]
    return c_dst
end
