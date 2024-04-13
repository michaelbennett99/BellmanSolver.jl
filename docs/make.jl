using Documenter, BellmanSolver

makedocs(
    sitename="BellmanSolver.jl",
    format = Documenter.HTML(; prettyurls = true),
    pages=[
        "index.md",
        "utils.md",
        "markov.md",
        "VFI.md",
        "egm.md"
    ]
)

deploydocs("github.com/michaelbennett99/BellmanSolver.jl")
