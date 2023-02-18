module Types

export Real_Vector, Real_Matrix, Real_3Array

Real_Vector = AbstractVector{<:Real}
Real_Matrix = AbstractMatrix{<:Real}
Real_3Array = AbstractArray{<:Real, 3}

end # module