using Interpolations, LinearAlgebra

export do_EGM

"""
    do_EGM(
        foc, env, lom, kp_grid, p_grid, trans_mat, β;
        tol=1e-6, max_iter=1000, kwargs...
    )

Solve a dynamic programming problem using the endogenous grid method.

# Arguments

- `foc::Function`: First order condition function
- `env::Function`: Envelope condition function
- `lom::Function`: Law of motion function
- `kp_grid::Vector{Float64}`: Grid points for the capital stock at time t+1
- `p_grid::Vector{Float64}`: Grid points for the price of capital
- `trans_mat::Matrix{Float64}`: Transition matrix
- `β::Real`: Discount factor
- `tol::Real`: Tolerance for convergence
- `max_iter::Integer`: Maximum number of iterations
- `kwargs...`: Additional keyword arguments for `foc`, `env`, and `lom`

# Returns

- `i_k_exog::Matrix{Float64}`: Policy function for the capital stock at time t
- `iter::Integer`: Number of iterations until convergence

# Throws

- `error`: If the algorithm does not converge after `max_iter` iterations
"""
function do_EGM(
        foc::Function, env::Function, lom::Function,
        kp_grid::Real_Vector, p_grid::Real_Vector, trans_mat::Real_Matrix, β::Real;
        tol::Real=1e-6, max_iter::Integer=1000, kwargs...
    )
    N_k = length(kp_grid)
    N_p = length(p_grid)

    # Make initial guess of the policy function
    i_k_exog = zeros(N_k, N_p)
    V = Matrix{Float64}(undef, N_k, N_p)
    for (j_kp, kp) ∈ enumerate(kp_grid), (j_p, p) ∈ enumerate(p_grid)
        V[j_kp, j_p] = env(kp, i_k_exog[j_kp, j_p], p; kwargs...)
    end

    # Create arrays we will use
    i_k_endog = Matrix{Float64}(undef, N_k, N_p)
    k = Matrix{Float64}(undef, N_k, N_p)
    i_k_exog_new = Matrix{Float64}(undef, N_k, N_p)

    for iter ∈ 1:max_iter
        # Update the guess of the policy function
        for (j_kp, kp) in enumerate(kp_grid), (j_p, p) in enumerate(p_grid)
            @views W_ij = β * (V[j_kp, :] ⋅ trans_mat[j_p, :])
            i_k_endog[j_kp, j_p] = foc(W_ij, p; kwargs...)
            @views k[j_kp, j_p] = lom(i_k_endog[j_kp, j_p], kp; kwargs...)
        end
        # Interpolate i/k onto the grid
        for j_p ∈ 1:N_p
            @views i_k_fn = linear_interpolation(
                k[:, j_p], i_k_endog[:, j_p], extrapolation_bc=Line())
            i_k_exog_new[:, j_p] = i_k_fn.(kp_grid)
        end
        # Check for convergence
        if maximum(abs.(i_k_exog_new - i_k_exog)) < tol
            return i_k_exog_new, iter
        # Update the value function
        else
            i_k_exog = copy(i_k_exog_new)
            for (i_kp, kp) in enumerate(kp_grid), (i_p, p) in enumerate(p_grid)
                V[i_kp, i_p] = env(kp, i_k_exog[i_kp, i_p], p; kwargs...)
            end
        end
    end
    throw(error("EGM did not converge after $max_iter iterations."))
end
