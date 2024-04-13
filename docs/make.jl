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

deploydocs(
    repo = "github.com/michaelbennett99/BellmanSolver.jl",
    devurl = "current"
)
