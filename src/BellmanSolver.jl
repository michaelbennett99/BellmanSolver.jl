module BellmanSolver

include("Utils.jl")
export make_k_grid


include("Markov.jl")
export tauchen, tauchen_unit_root, make_deterministic_chain


include("VFI.jl")
export do_VFI


include("EGM.jl")
export do_EGM

end # module