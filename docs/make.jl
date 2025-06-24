using Documenter
using TensorRenormalizationGroups

makedocs(
    sitename="TensorRenormalizationGroups.jl",
    authors="Jack Dunham",
    pages=[
        "Home" => "index.md",
        "Library" => [
            "Computing and contracting" => "renormalize.md",
            "Defining tensor networks" => "networks.md",
            "Algorithms" => "algorithms.md",
        ],
        "Index" => "_index.md"
    ]
)

deploydocs(
    repo="github.com/jack-dunham/TensorRenormalizationGroups.jl.git",
    devbranch="dev",
)
