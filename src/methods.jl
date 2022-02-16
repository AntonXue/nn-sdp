module Methods

using LinearAlgebra
using Parameters 
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork
using ..Qc

# Different kinds of inputs
abstract type InputConstraint end

# The set where {x : x1min <= x <= x1max}
@with_kw struct BoxInput <: InputConstraint
  x1min :: VecF64
  x1max :: VecF64
  @assert length(x1min) == length(x1max)
end

# The set where {x : Hx <= h}
@with_kw struct PolyInput <: InputConstraint
  H :: MatF64
  h :: VecF64
  @assert size(H)[1] == length(h)
end

# Safety constraints
# The set {x : [x; f(x); 1]' * S * [x; f(x); 1] <= 0
@with_kw struct SafetyConstraint
  S :: MatF64
end

# Some reachable set
abstract type ReachSet end

# Given a hyperplane normals such that each normalk' * x <= hk
@with_kw struct HplaneReachSet <: ReachSet
  normal :: VecF64
  @assert length(normal) >= 1
end

# Different problem specifications
abstract type Problem end

# Safety stuff
@with_kw struct SafetyProblem <: Problem
  nnet :: NeuralNetwork
  input :: InputConstraint
  output :: SafetyConstraint
  qcinfos :: Vector{QcInfo}
  @assert length(qcinfos) >= 1
end

# Reachability stuff
@with_kw struct ReachProblem <: Problem
  nnet :: NeuralNetwork
  input :: InputConstraint
  reach :: ReachSet
  qcinfos :: Vector{QcInfo}
  @assert length(qcinfos) >= 1
end

# The solution that is to be output by an algorithm
@with_kw struct ProblemSolution{A, B, C, D}
  objective_value :: A
  values :: B
  summary :: C
  termination_status :: D
  total_time :: Float64
  setup_time :: Float64
  solve_time :: Float64
end

# Calculate the sprasity pattern that we'll use
@with_kw struct SparsityPattern
  dummy :: Bool
end

include("methods/methods_common.jl")
include("methods/deep_sdp.jl")

export InputContraint, BoxInput, HplaneInput
export SafetyConstraint
export ReachSet, HplaneReachSet
export Problem, SafetyProblem, ReachProblem, ProblemSolution
export SparsityPattern, analyzeSparsity

export DeepSdpOptions
export run

end
