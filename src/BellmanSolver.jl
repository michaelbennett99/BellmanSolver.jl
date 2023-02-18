module BellmanSolver

include("Utils.jl")
using .Utils

export make_k_grid


include("Markov.jl")
using .Markov

export tauchen, tauchen_unit_root, make_deterministic_chain


include("VFI.jl")
using .VFI

export do_VFI


include("EGM.jl")
using .EGM

export do_EGM

end # module