using Documenter, BellmanSolver

makedocs(
    sitename="BellmanSolver.jl",
    format = Documenter.HTML(; prettyurls = true),
    pages=[
        "index.md"
    ]
)
