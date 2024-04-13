var documenterSearchIndex = {"docs":
[{"location":"egm/#Endogenous-Grid-Method","page":"Endogenous Grid Method","title":"Endogenous Grid Method","text":"","category":"section"},{"location":"egm/","page":"Endogenous Grid Method","title":"Endogenous Grid Method","text":"CurrentModule = BellmanSolver.EGM","category":"page"},{"location":"egm/","page":"Endogenous Grid Method","title":"Endogenous Grid Method","text":"Pages = [\"egm.md\"]","category":"page"},{"location":"egm/#Introduction","page":"Endogenous Grid Method","title":"Introduction","text":"","category":"section"},{"location":"egm/","page":"Endogenous Grid Method","title":"Endogenous Grid Method","text":"The Endogenous Grid Method (EGM) is a method to solve dynamic programming problems with one choice variable. It is particularly useful when the choice variable is continuous and the state space is high-dimensional. The EGM is a variant of the Value Function Iteration (VFI) algorithm that is more efficient in terms of computational time and memory usage.","category":"page"},{"location":"egm/#API","page":"Endogenous Grid Method","title":"API","text":"","category":"section"},{"location":"egm/","page":"Endogenous Grid Method","title":"Endogenous Grid Method","text":"Modules = [EGM]\nPrivate = false\nOrder   = [:type, :function]","category":"page"},{"location":"egm/#BellmanSolver.EGM.do_EGM-Tuple{Function, Function, Function, AbstractVector{<:Real}, AbstractVector{<:Real}, AbstractMatrix{<:Real}, Real}","page":"Endogenous Grid Method","title":"BellmanSolver.EGM.do_EGM","text":"do_EGM(\n    foc, env, lom, kp_grid, p_grid, trans_mat, β;\n    tol=1e-6, max_iter=1000, kwargs...\n)\n\nSolve a dynamic programming problem using the endogenous grid method.\n\nArguments\n\nfoc::Function: First order condition function\nenv::Function: Envelope condition function\nlom::Function: Law of motion function\nkp_grid::Vector{Float64}: Grid points for the capital stock at time t+1\np_grid::Vector{Float64}: Grid points for the price of capital\ntrans_mat::Matrix{Float64}: Transition matrix\nβ::Real: Discount factor\ntol::Real: Tolerance for convergence\nmax_iter::Integer: Maximum number of iterations\nkwargs...: Additional keyword arguments for foc, env, and lom\n\nReturns\n\ni_k_exog::Matrix{Float64}: Policy function for the capital stock at time t\niter::Integer: Number of iterations until convergence\n\nThrows\n\nerror: If the algorithm does not converge after max_iter iterations\n\n\n\n\n\n","category":"method"},{"location":"markov/#Markov-Chain","page":"Markov Chain","title":"Markov Chain","text":"","category":"section"},{"location":"markov/","page":"Markov Chain","title":"Markov Chain","text":"CurrentModule = BellmanSolver.Markov","category":"page"},{"location":"markov/","page":"Markov Chain","title":"Markov Chain","text":"Pages = [\"markov.md\"]","category":"page"},{"location":"markov/#Introduction","page":"Markov Chain","title":"Introduction","text":"","category":"section"},{"location":"markov/","page":"Markov Chain","title":"Markov Chain","text":"The markov chain model provides methods to set up discretised random processes for the random variable in dynamic programming problems. We use the tauchen method to create a markov chain from a continuous diffusion process.","category":"page"},{"location":"markov/#API","page":"Markov Chain","title":"API","text":"","category":"section"},{"location":"markov/","page":"Markov Chain","title":"Markov Chain","text":"Modules = [Markov]\nPrivate = false\nOrder   = [:type, :function]","category":"page"},{"location":"markov/#BellmanSolver.Markov.make_deterministic_chain-Tuple{Integer, Real, Real, Real}","page":"Markov Chain","title":"BellmanSolver.Markov.make_deterministic_chain","text":"make_deterministic_chain(N, min, max)\n\nMake a deterministic markov chain chain with N grid points between min and max.\n\nThis will be used to discretise the price state space in the deterministic case. While, strictly speaking, it might be both more readable and more performant to modify the value function instead, it is easier to code this method, and the performance of my algorithm is not currently a bottleneck.\n\nThis would be an identity matrix, but we need to account for the fact that η = 0.95, so prices will decay back to 1.\n\nArguments\n\nN::Int: Number of grid points\nmin::Real: Minimum value of the grid\nmax::Real: Maximum value of the grid\n\nReturns\n\ny_grid::Vector{Float64}: Grid points for the state space\ntrans_mat::Matrix{Float64}: Transition matrix for the markov chain\n\n\n\n\n\n","category":"method"},{"location":"markov/#BellmanSolver.Markov.tauchen-Tuple{Integer, Vararg{Real, 4}}","page":"Markov Chain","title":"BellmanSolver.Markov.tauchen","text":"tauchen(N, m, μ, σ, λ)\n\nTauchen's method for discretizing a state space for a AR(1) process.\n\nArguments\n\nN::Int: Number of grid points\nm::Real: Multiple of standard deviation for the grid\nμ::Real: Drift of the AR(1) process\nσ::Real: Standard deviation of the error term of the AR(1) process\nλ::Real: Persistence parameter of the AR(1) process\n\nReturns\n\ny_grid::Vector{Float64}: Grid points for the state space\ntrans_dict::Dict{Float64, Vector{Float64}}: Transition matrix for the AR(1)\n\nprocess\n\n\n\n\n\n","category":"method"},{"location":"markov/#BellmanSolver.Markov.tauchen_unit_root-Tuple{Integer, Real, Real}","page":"Markov Chain","title":"BellmanSolver.Markov.tauchen_unit_root","text":"tauchen_unit_root(N, m, σ)\n\nTauchen's method for discretizing a state space for a unit root AR(1) process.\n\nUsing a separate method as I am unsure how to handle the mean.\n\nArguments\n\nN::Int: Number of grid points\nm::Real: Multiple of standard deviation for the grid\nσ::Real: Standard deviation of the error term of the AR(1) process\n\nReturns\n\ny_grid::Vector{Float64}: Grid points for the state space\n\n\n\n\n\n","category":"method"},{"location":"VFI/#Value-Function-Iteration","page":"Value Function Iteration","title":"Value Function Iteration","text":"","category":"section"},{"location":"VFI/","page":"Value Function Iteration","title":"Value Function Iteration","text":"CurrentModule = BellmanSolver.VFI","category":"page"},{"location":"VFI/","page":"Value Function Iteration","title":"Value Function Iteration","text":"Pages = [\"vfi.md\"]","category":"page"},{"location":"VFI/#Introduction","page":"Value Function Iteration","title":"Introduction","text":"","category":"section"},{"location":"VFI/","page":"Value Function Iteration","title":"Value Function Iteration","text":"The Value Function Iteration (VFI) algorithm is a method to solve dynamic programming problems. It is particularly useful when the state space is low-dimensional. The VFI algorithm is a brute-force method that iteratively applies the Bellman operator to the value function until convergence. The VFI algorithm is computationally expensive because it requires solving the Bellman equation at each iteration. However, it is a simple and robust method that can be used as a benchmark for more sophisticated algorithms.","category":"page"},{"location":"VFI/#API","page":"Value Function Iteration","title":"API","text":"","category":"section"},{"location":"VFI/","page":"Value Function Iteration","title":"Value Function Iteration","text":"Modules = [VFI]\nPrivate = false\nOrder   = [:type, :function]","category":"page"},{"location":"VFI/#BellmanSolver.VFI.do_VFI-Tuple{Function, AbstractVector{<:Real}, AbstractVector{<:Real}, AbstractMatrix{<:Real}, Real}","page":"Value Function Iteration","title":"BellmanSolver.VFI.do_VFI","text":"do_VFI(\n    flow_value, k_grid, p_grid, trans_mat, β;\n    tol=1e-6, max_iter=1000, kwargs...\n)\n\nDo value function iteration for the case where we optimise over one variable, and there is one stochastic variable in the value function. The stochastic variable is represented by a markov process on a grid.\n\nArguments\n\nflow_value::Function: Function to compute the flow value\nk_grid::Vector{Float64}: Grid points for the capital stock\np_grid::Vector{Float64}: Grid points for the price of capital\ntrans_mat::Array{Float64, 2}: Transition matrix for the price process\nβ::Real: Discount factor\ntol::Real=1e-6: Tolerance for convergence\nmax_iter::Int=1000: Maximum number of iterations\nkwargs...: Additional keyword arguments for flow_value\n\nReturns\n\nk_grid::Vector{Float64}: Grid points for the capital stock\np_grid::Vector{Float64}: Grid points for the price of capital\nKp_mat::Array{Float64, 2}: Policy function for the capital stock at time t+1\nV::Array{Float64, 2}: Value function\n\n\n\n\n\n","category":"method"},{"location":"VFI/#BellmanSolver.VFI.do_VFI-Tuple{Function, AbstractVector{<:Real}, Real, NumericalMethods.Interp.AbstractInterpolator}","page":"Value Function Iteration","title":"BellmanSolver.VFI.do_VFI","text":"do_VFI(flow_value, k_grid, β, interp; tol=1e-6, max_iter=1000, kwargs...)\n\nDo Value Function Iteration for the case where we optimise over one variable and there are no stochastic elements, using interpolation to optimise over an interval for each grid point.\n\nArguments\n\nflow_value::Function: Function to calculate the flow value\nk_grid::Vector{Float64}: Grid points for the choice variable\nβ::Real: Discount factor\ninterp::AbstractInterpolator: Interpolator to use\ntol::Real=1e-6: Tolerance for convergence\nmax_iter::Int=1000: Maximum number of iterations\nkwargs...: Keyword arguments to pass to flow_value. Should be parameters   of flow_value\n\nReturns\n\nk_grid::Vector{Float64}: Grid points for the choice variable\nkp_vct::Vector{Float64}: Optimal choice\nV::Vector{Float64}: Value function\n\n\n\n\n\n","category":"method"},{"location":"VFI/#BellmanSolver.VFI.do_VFI-Tuple{Function, AbstractVector{<:Real}, Real}","page":"Value Function Iteration","title":"BellmanSolver.VFI.do_VFI","text":"do_VFI(flow_value, k_grid, β; tol=1e-6, max_iter=1000, kwargs...)\n\nDo value function iteration for the case where we optimise over one variable, and there are no stochastic elements.\n\nArguments\n\nflow_value::Function: Function to calculate the flow value\nk_grid::Vector{Float64}: Grid points for the choice variable\nβ::Real: Discount factor\ntol::Real=1e-6: Tolerance for convergence\nmax_iter::Int=1000: Maximum number of iterations\nkwargs...: Keyword arguments to pass to flow_value. Should be parameters   of flow_value\n\nReturns\n\nk_grid::Vector{Float64}: Grid points for the choice variable\nkp_vct::Vector{Float64}: Optimal choice\nV::Vector{Float64}: Value function\n\n\n\n\n\n","category":"method"},{"location":"#BellmanSolver.jl-Documentation","page":"BellmanSolver.jl Documentation","title":"BellmanSolver.jl Documentation","text":"","category":"section"},{"location":"","page":"BellmanSolver.jl Documentation","title":"BellmanSolver.jl Documentation","text":"Pages = [\"utils.md\", \"markov.md\", \"VFI.md\", \"egm.md\"]\nDepth = 1","category":"page"},{"location":"#Welcome-to-BellmanSolver.jl","page":"BellmanSolver.jl Documentation","title":"Welcome to BellmanSolver.jl","text":"","category":"section"},{"location":"","page":"BellmanSolver.jl Documentation","title":"BellmanSolver.jl Documentation","text":"This package provides methods to efficiently set up and solve dynamic programming problems of one choice variable using Value Function Iteration and the Endogenous Grid Method.","category":"page"},{"location":"#Index","page":"BellmanSolver.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"BellmanSolver.jl Documentation","title":"BellmanSolver.jl Documentation","text":"","category":"page"},{"location":"utils/#Utils","page":"Utils","title":"Utils","text":"","category":"section"},{"location":"utils/","page":"Utils","title":"Utils","text":"CurrentModule = BellmanSolver.Utils","category":"page"},{"location":"utils/","page":"Utils","title":"Utils","text":"Pages = [\"utils.md\"]","category":"page"},{"location":"utils/#Introduction","page":"Utils","title":"Introduction","text":"","category":"section"},{"location":"utils/","page":"Utils","title":"Utils","text":"This module provides utilities for setting up dynamic programs.","category":"page"},{"location":"utils/#API","page":"Utils","title":"API","text":"","category":"section"},{"location":"utils/","page":"Utils","title":"Utils","text":"Modules = [Utils]\nPrivate = false\nOrder   = [:type, :function]","category":"page"},{"location":"utils/#BellmanSolver.Utils.make_k_grid-Tuple{Real, Real, Integer}","page":"Utils","title":"BellmanSolver.Utils.make_k_grid","text":"make_k_grid(min, max, N)\n\nMake a grid for the capital stock.\n\nArguments\n\nmin::Real: Minimum value of the grid\nmax::Real: Maximum value of the grid\nN::Int: Number of grid points\n\nReturns\n\nk_grid::Vector{Float64}: Grid points for the capital stock\n\n\n\n\n\n","category":"method"}]
}
