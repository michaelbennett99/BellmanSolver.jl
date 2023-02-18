module BellmanSolver

include("Utils.jl")
include("Markov.jl")
include("VFI.jl")
include("EGM.jl")

export make_k_grid
export tauchen, tauchen_unit_root, make_deterministic_chain
export do_VFI
export do_EGM

end # module