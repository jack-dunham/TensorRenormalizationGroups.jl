"""
$(TYPEDEF)

Abstract supertype of all coarse-graining algorithms.
"""
abstract type AbstractGrainingAlgorithm <: AbstractRenormalizationAlgorithm end
"""
$(TYPEDEF)

Abstract supertype of all coarse-graining runtime state objects.
"""
abstract type AbstractGrainingRuntime <: AbstractRenormalizationRuntime end
