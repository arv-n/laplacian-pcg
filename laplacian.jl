mutable struct LaplacianPreconditioner{T, S} 
    L::S
    M::Int
    function LaplacianPreconditioner(A::SparseMatrixCSC{T}) where {T}
        L = A
        return new{eltype(L), typeof(L)}(L)
    end
end

@inline function LinearAlgebra.ldiv!(y, P::LaplacianPreconditioner, b) 
    y = P.L\b
    return y;
end
@inline LinearAlgebra.ldiv!(P::LaplacianPreconditioner,b) = ldiv!(b,P,b)


