# Solves AL * C = C * A (= AC)

function lgsolve(x, T)
    y = similar(x)
    @tensoropt y[ur dr] = x[ul dl] * T[dr dl; ur ul]
    return y
end

function leftgauge!(
    AL::CircularVector{<:AbsTen{N,2,S}},
    L::CircularVector{<:AbsTen{0,2,S}},
    A::CircularVector{<:AbsTen{N,2,S}};
    tol=1e-14,
    maxiter=100,
    verbose=false,
) where {S,N}
    # verbose = true
    numsites = length(A)
    r = 1:numsites

    ϵ = 2 * tol
    numiter = 1

    normalize!(L[end])

    λ = zeros(numsites)
    ϵ = zeros(numsites)

    ϵ[end] = 2 * tol

    T = multi_transfer(A, A) # dr dl, ur ul -> ul dl, ur dr

    axpby!(1//2, T', 1//2, T)

    # Tp = permute(T, (4, 2), (3, 1))

    λs, Ls, info = KrylovKit.eigsolve(L[end], 1, :LM; ishermitian=false, eager=true) do x0
        x1 = similar(x0)
        @tensoropt x1[ur dr] = x0[ul dl] * T[dr dl; ur ul]
    end

    Lp = permute(Ls[1], ((1,), (2,)))

    @debug "MPS fixed point info:" λ = λs[1] info = info hermiticity = norm(Lp - Lp')
    # Lp = 1 / 2 * (Lp + Lp')

    # Eigenvector can be α * v, so choose α s.t. v positive
    # Lp = sqrt(Lp * Lp)

    # D, V = eigh(Lp)

    # L[end] = permute(V * sqrt(D) * V', (), (1, 2))
    permute!(L[end], sqrt(Lp * Lp), ((), (1, 2)))

    for x in r
        Lold = copy(L[x])

        _, Li = leftdecomp!(AL[x], copy(L[x - 1]), A[x])

        copy!(L[x], Li)
        normalize!(L[x])

        ϵ[x] = norm(Lold - L[x])
    end

    verbose && @info "MPS gauging error: $ϵ"

    while numiter < maxiter && max(ϵ...) > tol
        for x in r
            Lold = copy(L[x])

            _, Li = leftdecomp!(AL[x], copy(L[x - 1]), A[x])

            copy!(L[x], Li)
            normalize!(L[x])

            ϵ[x] = norm(Lold - L[x])
            @debug "$ϵ"
        end

        verbose && @info "MPS gauging error: $ϵ"
        numiter += 1
    end
    return AL, L, λ
end

function rightgauge!(
    AR::CircularVector{<:AbsTen{N,2,S}},
    R::CircularVector{<:AbsTen{0,2,S}},
    A::CircularVector{<:AbsTen{N,2,S}};
    kwargs...,
) where {S,N}
    # Permute left and right virtual bonds bonds
    AL = swapvirtual.(AR)
    L = swapvirtual.(R)
    A = swapvirtual.(A)

    # Run the gauging algorithm
    _, _, λ = leftgauge!(AL, L, A; kwargs...)

    # Permute left and right bonds
    permutedom!.(AR, AL, Ref((2, 1)))
    permutedom!.(R, L, Ref((2, 1)))
    return AR, R, λ
end

# AL[x] <- L[x - 1] * A[x]
function leftdecomp!(AL, L, A::AbsTen{N,2}) where {N}
    # Overwrite AL
    mulbond!(AL, L, A)
    AL_p = permute(AL, (tuple(1:N..., N + 2)::NTuple{N + 1,Int}, (N + 1,)))
    temp_AL, temp_L = leftorth!(AL_p)
    permute!(AL, temp_AL, (Tuple(1:N)::NTuple{N,Int}, (N + 2, N + 1)))
    permute!(L, temp_L, ((), (2, 1)))
    return AL, L
end

# Find the mixedguage of uniform MPS A, writing result into AL,C,AR
function mixedgauge!(AL, C, AR, A; verbose=false, kwargs...)
    for y in axes(A, 2)
        verbose && @info "Gauging right..."
        rightgauge!(AR[:, y], C[:, y], A[:, y]; verbose=verbose, kwargs...)
        verbose && @info "Gauging left..."
        leftgauge!(AL[:, y], C[:, y], AR[:, y]; verbose=verbose, kwargs...)
    end
    # currently not working 
    diagonalise!(AL, C, AR)
    return AL, C, AR
end

# Diagonalise the bond matrices using the singular value decomposition.
function diagonalise!(
    AL::AbstractUnitCell{G,<:AbstractTensorMap{S}},
    C::AbstractUnitCell{G,<:AbstractTensorMap{S}},
    AR::AbstractUnitCell{G,<:AbstractTensorMap{S}},
) where {G,S}
    U = similar(C)
    V = similar(C)

    for y in axes(C, 2)
        for x in axes(C, 1)
            U[x, y], C[x, y], V[x, y] = _tsvd(C[x, y])
        end
    end
    gauge!(AL, U)
    gauge!(V, AR)
    return AL, C, AR
end
function diagonalise!(mps::MPS)
    AL, C, AR, AC = mps
    diagonalise!(AL, C, AR)
    centraltensor!(AC, AL, C)
    return mps
end

function _tsvd(t)
    u, s, v = tsvd(t, (2,), (1,))
    up = permute(u, (), (2, 1))
    sp = permute(s, (), (2, 1))
    vp = permute(v, (), (2, 1))
    return up, sp, vp
end

# Vectorised form of a gauge transformation. That is, A[x] <- U'[x-1]*A[x]*U[x]
# AL[x]*C[x] -> AL[x]*U[x]*C[x]*V[x]
# C[x - 1]*AR[x] -> U[x - 1]*C[x - 1]*V[x - 1]*AR[x]
function gauge!(
    A::AbstractUnitCell{G,<:AbstractTensorMap{<:Number,S}},
    U::AbstractUnitCell{G,<:AbstractTensorMap{<:Number,S,0,2}},
) where {G,S}
    UU = broadcast(u -> TensorMap(cj(permutedom(u, (2, 1))).data, space(u)), U)

    centraltensor!(A, UU, centraltensor(A, U))
    return A
end
function gauge!(
    U::AbstractUnitCell{G,<:AbstractTensorMap{<:Number,S,0,2}},
    A::AbstractUnitCell{G,<:AbstractTensorMap{<:Number,S}},
) where {G,S}
    UU = broadcast(u -> TensorMap(cj(permutedom(u, (2, 1))).data, space(u)), U)

    centraltensor!(A, U, centraltensor(A, UU))
    return A
end
function mps_transfer(a::AbsTen{2,2}, al::AbsTen{2,2})
    @tensoropt T[dr dl; ur ul] := a[p1 p2; ur ul] * (al')[dr dl; p1 p2]
    return T
end
function mps_transfer(a::AbsTen{1,2}, al::AbsTen{1,2})
    @tensoropt T[dr dl; ur ul] := a[p1; ur ul] * (al')[dr dl; p1]
    return T
end

function compose_transfer(T1, T2)
    @tensoropt T[dr dl; ur ul] := T1[dr d; ur u] * T2[d dl; u ul]
    return T
end

function multi_transfer(A::AbstractArray, AL::AbstractArray)
    T = mps_transfer(A[1], A[1])
    for i in axes(A, 1)
        if i == 1
            continue
        end
        Ti = mps_transfer(A[i], AL[i])
        T = compose_transfer(T, Ti)
    end
    # T = T / norm(T)
    return T
end
