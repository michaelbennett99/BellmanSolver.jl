using LinearAlgebra, Interpolation, Optim

export do_VFI

"""
    make_flow_value_mat(flow_value, k_grid; kwargs...)

Make a matrix of flow values for all combinations of the state and choice
variable, assuming that these inhabit the same grid.

# Arguments

- `flow_value::Function`: Function that takes the choice variable and the state
    variable as arguments, and returns the flow value.
- `k_grid::Vector{Float64}`: Grid points for the choice variable
- `kwargs...`: Keyword arguments to be passed to `flow_value`

# Returns

- `flow_val_mat::Matrix{Float64}`: Matrix of flow values
"""
function make_flow_value_mat(
        flow_value::Function, k_grid::Real_Vector; kwargs...
    )
    k_N = length(k_grid)
    flow_val_mat = Matrix{Float64}(undef, k_N, k_N)
    for (i_k, k) ∈ enumerate(k_grid)
        for (i_kp, kp) ∈ enumerate(k_grid)
                flow_val_mat[i_k, i_kp] = flow_value(k, kp; kwargs...)
            end
        end
    return flow_val_mat
end


"""
    make_flow_value_mat(flow_value, k_grid, kp_grid, p_grid; kwargs...)

Make a matrix of flow values for all combinations of the choice variable, as
well as the deterministic and stochastic state variables.

# Arguments

- `flow_value::Function`: Function to calculate the flow value
- `k_grid::Vector{Float64}`: Grid points for 
- `kp_grid::Vector{Float64}`: Grid points for capital stock next period
- `p_grid::Vector{Float64}`: Grid points for price
- `kwargs...`: Keyword arguments to pass to `flow_value`. Should be parameters
    of flow_value

# Returns

- `flow_val_mat::Array{Float64, 3}`: 3-array of flow values
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
    do_VFI(flow_value, k_grid, β; tol=1e-6, max_iter=1000, kwargs...)

Do value function iteration for the case where we optimise over one variable,
and there are no stochastic elements.

# Arguments

- `flow_value::Function`: Function to calculate the flow value
- `k_grid::Vector{Float64}`: Grid points for the choice variable
- `β::Real`: Discount factor
- `tol::Real=1e-6`: Tolerance for convergence
- `max_iter::Int=1000`: Maximum number of iterations
- `kwargs...`: Keyword arguments to pass to `flow_value`. Should be parameters
    of flow_value

# Returns

- `k_grid::Vector{Float64}`: Grid points for the choice variable
- `kp_vct::Vector{Float64}`: Optimal choice
- `V::Vector{Float64}`: Value function
"""
function do_VFI(
        flow_value::Function, k_grid::Real_Vector, β::Real;
        tol::Real=1e-6, max_iter::Integer=1000, kwargs...
    )
    println("Starting Value Function Iteration...")

    k_N = length(k_grid)
    V = zeros(k_N)
    kp_vct = Vector{Float64}(undef, k_N)

    println("Making flow value matrix...")

    flow_val_mat = make_flow_value_mat(flow_value, k_grid; kwargs...)

    println("Starting iteration...")

    diff = 1
    iter = 0
    while diff > tol
        val_vct = Vector{Float64}(undef, k_N)
        for i_k ∈ 1:length(k_grid)
            val = -Inf
            val_kp = NaN
            for (i_kp, kp) ∈ enumerate(k_grid)
                @views candidate_val = flow_val_mat[i_k, i_kp] + β * V[i_kp]
                if candidate_val > val
                    val = candidate_val
                    val_kp = kp
                end
            end
            val_vct[i_k] = val
            kp_vct[i_k] = val_kp
        end
        diff = maximum(abs.(V - val_vct))
        V = val_vct
        iter += 1
        if iter % 10 == 0
            println("Iteration $iter finished, Diff: $diff.")
        end
        if iter > max_iter
            break
        end
    end
    return k_grid, kp_vct, V
end

function do_VFI(
        flow_value::Function, k_grid::Real_Vector, β::Real, interp::Function;
        tol::Real=1e-6, max_iter::Integer=1000, kwargs...
    )
    println("Starting Value Function Iteration...")

    k_N = length(k_grid)
    V = zeros(k_N)
    V_i = zeros(k_N)
    kp_vct = Vector{Float64}(undef, k_N)

    println("Making flow value matrix...")

    flow_val_mat = make_flow_value_mat(flow_value, k_grid; kwargs...)

    println("Starting iteration...")

    diff = 1
    iter = 0
    while diff > tol
        val_vct = Vector{Float64}(undef, k_N)
        for i_k ∈ 1:k_N
            for i_kp ∈ 1:k_N
                @views V_i[i_kp] = flow_val_mat[i_k, i_kp] + β * V[i_kp]
            end
            feasible = .!isinf.(V_i)
            f_k = k_grid[feasible]
            f_V = V_i[feasible]
            V_i_fn = interp(f_k, f_V)
            @views result = Optim.maximize(V_i_fn, f_k[1], f_k[end], Brent())
            if ! Optim.converged(result)
                error("Optimization did not converge.")
            end
            kp_max = Optim.maximizer(result)
            val_vct[i_k] = Optim.maximum(result)
            kp_vct[i_k] = kp_max
        end
        diff = maximum(abs.(V - val_vct))
        V = val_vct
        iter += 1
        if iter % 10 == 0
            println("Iteration $iter finished, Diff: $diff.")
        end
        if iter > max_iter
            break
        end
    end
    return k_grid, kp_vct, V
end


"""
    do_VFI(
        flow_value, k_grid, p_grid, trans_mat, β;
        tol=1e-6, max_iter=1000, kwargs...
    )

Do value function iteration for the case where we optimise over one variable,
and there is one stochastic variable in the value function. The stochastic
variable is represented by a markov process on a grid.

# Arguments

- `flow_value::Function`: Function to compute the flow value
- `k_grid::Vector{Float64}`: Grid points for the capital stock
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `trans_mat::Array{Float64, 2}`: Transition matrix for the price process
- `β::Real`: Discount factor
- `tol::Real=1e-6`: Tolerance for convergence
- `max_iter::Int=1000`: Maximum number of iterations
- `kwargs...`: Additional keyword arguments for flow_value

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