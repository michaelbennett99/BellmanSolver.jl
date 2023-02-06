module BellmanSolver

using Distributions, StatsBase, LinearAlgebra

export tauchen, tauchen_unit_root, make_deterministic_chain, single_price_chain
export make_k_grid, make_kp_grid
export do_VFI

Real_Vector = AbstractVector{<:Real}
Real_Matrix = AbstractMatrix{<:Real}
Real_3Array = AbstractArray{<:Real, 3}

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

function single_price_chain(y_val::Real)
    trans_mat = ones((1, 1))
    y_grid = [y_val]
    return y_grid, trans_mat
end

"""
    make_flow_value_mat(k_grid, kp_grid, p_grid, w, α, δ)

Make a matrix of flow values for all combinations of capital stock and price.

# Arguments

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `kp_grid::Vector{Float64}`: Grid points for the capital stock at time t+1
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `w::Real`: Wage rate at time t
- `α::Real`: Productivity parameter
- `δ::Real`: Depreciation rate

# Returns

- `flow_val_mat::Array{Float64, 3}`: Matrix of flow values
"""
function make_flow_value_mat(
        flow_value::Function,
        k_grid::Real_Vector, kp_grid::Real_Vector, p_grid::Real_Vector;
        kwargs...
    )
    k_N = length(k_grid)
    kp_N = length(kp_grid)
    p_N = length(p_grid)
    flow_val_mat = Array{Float64, 3}(undef, k_N, kp_N, p_N)
    for (i_k, k) ∈ enumerate(k_grid)
        for (i_kp, kp) ∈ enumerate(kp_grid)
            for (i_p, p) ∈ enumerate(p_grid)
                flow_val_mat[i_k, i_kp, i_p] = flow_value(k, kp, p; kwargs...)
            end
        end
    end
    return flow_val_mat
end

"""
    value_function(flow_val_mat, V, trans_mat, i_k, i_p, i_kp, β)

Compute the value function based on a matrix of flow values, an old value
function and a transition matrix.

# Arguments

- `flow_value_mat::Array{Float64, 3}`: Matrix of flow values
- `V::Array{Float64, 2}`: Old value function
- `trans_mat::Array{Float64, 2}`: Transition matrix for the price process
- `i_k::Int`: Index of the capital stock
- `i_p::Int`: Index of the price of capital
- `i_kp::Int`: Index of the capital stock at time t+1
- `β::Real`: Discount factor

# Returns

- `value::Float64`: Value function
"""
function value_function(
        flow_value_mat::Real_3Array, V::Real_Matrix, trans_mat::Real_Matrix,
        i_k::Integer, i_p::Integer, i_kp::Integer, β::Real
    )
    @views flow_val = flow_value_mat[i_k, i_kp, i_p]
    @views ECont = V[i_kp, :] ⋅ trans_mat[i_p, :]
    return flow_val + β * ECont
end

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

"""
    make_kp_grid(k_grid, δ)

Make a grid for the capital stock at time t+1, where inaction is always an
option.

# Arguments

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `δ::Real`: Depreciation rate

# Returns

- `kp_grid::Vector{Float64}`: Grid points for the capital stock at time t+1
- `k_kp_mapping::Dict{Int, Int}`: Mapping from the index of the capital stock
    to the index of the capital stock at time t+1
"""
function make_kp_grid(k_grid::Real_Vector, δ::Real)
    new_values = k_grid .* (1 - δ)
    kp_grid_full = unique(sort(cat(k_grid, new_values, dims=1)))
    k_kp_mapping = Dict{Int, Int}()
    for (i, k) ∈ enumerate(k_grid)
        i_kp = findfirst(kp_grid_full .== k)
        k_kp_mapping[i] = i_kp
    end
    return kp_grid_full, k_kp_mapping
end

"""
    do_VFI(
        k_grid, p_grid, trans_mat, α, δ, r, g, w;
        tol=1e-6, max_iter=1000
    )

Run value function iteration.

# Arguments

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `trans_mat::Array{Float64, 2}`: Transition matrix for the price process
- `α::Real`: Productivity parameter
- `δ::Real`: Depreciation rate
- `r::Real`: Interest rate
- `g::Real`: Growth rate of the wage rate
- `w::Real`: Wage rate at time t
- `tol::Real`: Tolerance for convergence
- `max_iter::Int`: Maximum number of iterations

# Returns

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `Kp_mat::Array{Float64, 2}`: Policy function for the capital stock at time t+1
- `V::Array{Float64, 2}`: Value function
"""
function do_VFI(
        flow_value::Function, k_grid::Real_Vector, p_grid::Real_Vector,
        trans_mat::Real_Matrix, β::Real;
        tol::Real=1e-6, max_iter::Integer=1000, kwargs...
    )
    println("Starting Value Function Iteration...")

    Kp_grid = k_grid

    k_N = length(k_grid)
    p_N = length(p_grid)

    V = zeros(k_N, p_N)

    Kp_mat = Matrix{Float64}(undef, k_N, p_N)

    println("Making flow value matrix...")

    flow_val_mat = make_flow_value_mat(
        flow_value, k_grid, Kp_grid, p_grid; kwargs...
    )

    println("Starting iteration...")

    diff = 1
    iter = 0
    while diff > tol
        val_mat = Matrix{Float64}(undef, k_N, p_N)
        for i_k ∈ 1:length(k_grid), i_p ∈ 1:length(p_grid)
            val = -Inf
            val_kp = NaN
            for (i_kp, kp) ∈ enumerate(Kp_grid)
                candidate_val = value_function(
                    flow_val_mat, V, trans_mat, i_k, i_p, i_kp, β
                )
                if candidate_val > val
                    val = candidate_val
                    val_kp = kp
                end
            end
            val_mat[i_k, i_p] = val
            Kp_mat[i_k, i_p] = val_kp
        end
        diff = maximum(abs.(V - val_mat))
        V = val_mat
        iter += 1
        if iter % 10 == 0
            println("Iteration $iter finished, Diff: $diff.")
        end
        if iter > max_iter
            break
        end
    end
    return k_grid, p_grid, Kp_mat, V
end

"""
    do_VFI(
        flow_value, k_grid, kp_grid, p_grid, trans_mat, β;
        tol=1e-6, max_iter=1000, kwargs...
    )

Run value function iteration.

# Arguments

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `trans_mat::Array{Float64, 2}`: Transition matrix for the price process
- `α::Real`: Productivity parameter
- `δ::Real`: Depreciation rate
- `r::Real`: Interest rate
- `g::Real`: Growth rate of the wage rate
- `w::Real`: Wage rate at time t
- `tol::Real`: Tolerance for convergence
- `max_iter::Int`: Maximum number of iterations

# Returns

- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `Kp_mat::Array{Float64, 2}`: Policy function for the capital stock at time t+1
- `V::Array{Float64, 2}`: Value function
"""
function do_VFI(
        flow_value::Function,
        k_grid::Real_Vector, kp_grid::Real_Vector, p_grid::Real_Vector,
        trans_mat::Real_Matrix, β::Real;
        tol::Real=1e-6, max_iter::Integer=1000, kwargs...
    )
    println("Starting Value Function Iteration...")

    k_N = length(k_grid)
    p_N = length(p_grid)
    kp_N = length(kp_grid)

    V = zeros(kp_N, p_N)

    kp_mat = Matrix{Float64}(undef, kp_N, p_N)

    println("Making flow value matrix...")

    flow_val_mat = make_flow_value_mat(
        flow_value, k_grid, kp_grid, p_grid; kwargs...
    )

    println("Starting iteration...")

    diff = 1
    iter = 0
    while diff > tol
        val_mat = Matrix{Float64}(undef, k_N, p_N)
        for i_k ∈ 1:length(k_grid), i_p ∈ 1:length(p_grid)
            val = -Inf
            val_kp = NaN
            for (i_kp, kp) ∈ enumerate(kp_grid)
                candidate_val = value_function(
                    flow_val_mat, V, trans_mat, i_k, i_p, i_kp, β
                )
                if candidate_val > val
                    val = candidate_val
                    val_kp = kp
                end
            end
            val_mat[i_k, i_p] = val
            kp_mat[i_k, i_p] = val_kp
        end
        diff = maximum(abs.(V - val_mat))
        V = val_mat
        iter += 1
        if iter % 10 == 0
            println("Iteration $iter finished, Diff: $diff.")
        end
        if iter > max_iter
            break
        end
    end
    return k_grid, p_grid, kp_mat, V
end

end