# Value Function Iteration

```@meta
CurrentModule = BellmanSolver.VFI
```

```@contents
Pages = ["vfi.md"]
```

## Introduction

The Value Function Iteration (VFI) algorithm is a method to solve dynamic programming problems. It is particularly useful when the state space is low-dimensional. The VFI algorithm is a brute-force method that iteratively applies the Bellman operator to the value function until convergence. The VFI algorithm is computationally expensive because it requires solving the Bellman equation at each iteration. However, it is a simple and robust method that can be used as a benchmark for more sophisticated algorithms.

## API

```@autodocs
Modules = [VFI]
Private = false
Order   = [:type, :function]
```
