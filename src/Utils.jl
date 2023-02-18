"""
    make_k_grid(min, max, N)

Make a grid for the capital stock.

# Arguments

- `min::Real`: Minimum value of the grid
- `max::Real`: Maximum value of the grid
- `N::Int`: Number of grid points

# Returns

- `k_grid::Vector{Float64}`: Grid points for the capital stock
"""
function make_k_grid(min::Real, max::Real, N::Integer)
    return collect(Float64, range(min, max, N))
end