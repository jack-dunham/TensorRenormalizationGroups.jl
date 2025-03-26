abstract type AbstractBoundaryAlgorithm <: AbstractAlgorithm end
abstract type AbstractBoundaryRuntime <: AbstractRuntime end

function boundaryerror!(S_old::AbstractMatrix, C_new::AbstractMatrix)
    S_new = boundaryerror.(C_new)
    if all(space.(S_new) == space.(S_old))
        err = @. norm(S_old - S_new)
    else
        err = broadcast(S_old, S_new) do s_old, s_new
            wold = diag(convert(Array, s_old))
            wnew = diag(convert(Array, s_new))

            wold = wold / sum(wold)
            wnew = wnew / sum(wnew)

            entf = λ -> λ ≈ 0 ? 0 : -λ * log(λ)

            eold = sum(entf, wold) / log(length(wold))
            enew = sum(entf, wnew) / log(length(wnew))

            rv = abs(eold - enew)

            return rv
        end
    end
    S_old .= S_new
    return err
end

function boundaryerror(c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, ((1,), (2,)))
    normalize!(s_new)
    return s_new
end
