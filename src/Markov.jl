using Distributions

export tauchen, tauchen_unit_root, make_deterministic_chain

"""
    tauchen(N, m, μ, σ, λ)

Tauchen's method for discretizing a state space for a AR(1) process.

# Arguments

- `N::Int`: Number of grid points
- `m::Real`: Multiple of standard deviation for the grid
- `μ::Real`: Drift of the AR(1) process
- `σ::Real`: Standard deviation of the error term of the AR(1) process
- `λ::Real`: Persistence parameter of the AR(1) process

# Returns

- `y_grid::Vector{Float64}`: Grid points for the state space
- `trans_dict::Dict{Float64, Vector{Float64}}`: Transition matrix for the AR(1)
process
"""
function tauchen(N::Integer, m::Real, μ::Real, σ::Real, λ::Real)
    a = (1 - λ) * μ
    σ_y = σ / sqrt((1 - λ^2))

    y_1 = -(m * σ_y)
    y_N = m * σ_y
    y_grid = collect(range(y_1, y_N, N))
    y_grid .+= μ

    w = y_grid[2] - y_grid[1]

    trans_mat = Matrix{Float64}(undef, N, N)
    for (i, y_i) in enumerate(y_grid)
        for (j, y_j) in enumerate(y_grid)
            p_ij_1 = cdf(Normal(), (y_j + w/2 - λ * y_i - a) / σ)
            p_ij_2 = cdf(Normal(), (y_j - w/2 - λ * y_i - a) / σ)
            if j == 1
                p_ij = p_ij_1
            elseif j == N
                p_ij = 1 - p_ij_2
            else
                p_ij = p_ij_1 - p_ij_2
            end
            trans_mat[i, j] = p_ij
        end
    end
    return y_grid, trans_mat
end


"""
    tauchen_unit_root(N, m, σ)

Tauchen's method for discretizing a state space for a unit root AR(1) process.

Using a separate method as I am unsure how to handle the mean.

# Arguments

- `N::Int`: Number of grid points
- `m::Real`: Multiple of standard deviation for the grid
- `σ::Real`: Standard deviation of the error term of the AR(1) process

# Returns

- `y_grid::Vector{Float64}`: Grid points for the state space
"""
function tauchen_unit_root(N::Integer, m::Real, σ::Real)
    y_1 = -(m * σ)
    y_N = m * σ
    y_grid = collect(range(y_1, y_N, N))

    w = y_grid[2] - y_grid[1]

    trans_mat = Matrix{Float64}(undef, N, N)
    for (i, y_i) in enumerate(y_grid)
        for (j, y_j) in enumerate(y_grid)
            p_ij_1 = cdf(Normal(), (y_j - y_i + w/2) / σ)
            p_ij_2 = cdf(Normal(), (y_j - y_i - w/2) / σ)
            if j == 1
                p_ij = p_ij_1
            elseif j == N
                p_ij = 1 - p_ij_2
            else
                p_ij = p_ij_1 - p_ij_2
            end
            trans_mat[i, j] = p_ij
        end
    end
    return y_grid, trans_mat
end


"""
    make_deterministic_chain(N, min, max)

Make a deterministic markov chain chain with N grid points between min and max.

This will be used to discretise the price state space in the deterministic case.
While, strictly speaking, it might be both more readable and more performant to
modify the value function instead, it is easier to code this method, and the
performance of my algorithm is not currently a bottleneck.

This would be an identity matrix, but we need to account for the fact that η =
0.95, so prices will decay back to 1.

# Arguments

- `N::Int`: Number of grid points
- `min::Real`: Minimum value of the grid
- `max::Real`: Maximum value of the grid

# Returns

- `y_grid::Vector{Float64}`: Grid points for the state space
- `trans_mat::Matrix{Float64}`: Transition matrix for the markov chain
"""
function make_deterministic_chain(N::Integer, min::Real, max::Real, η::Real)
    y_grid = exp.(collect(range(min, max, N)))
    trans_mat = zeros(N, N)
    for i ∈ 1:N
        next_y = (y_grid[i]) ^ η
        i2 = findfirst(y_grid .> next_y)
        coef = (next_y - y_grid[i2 - 1]) / (y_grid[i2] - y_grid[i2-1])
        trans_mat[i, i2] = coef
        trans_mat[i, i2 - 1] = 1 - coef
    end
    return log.(y_grid), trans_mat
end
