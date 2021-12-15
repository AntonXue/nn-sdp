# Define a bunch of types we would like to have around
module Header

using Parameters

# Different types of networks
abstract type NetworkType end
struct ReluNetwork <: NetworkType end
struct TanhNetwork <: NetworkType end

# Generic neural network supertype
abstract type NeuralNetwork end

# Parameters needed to define the feed-forward network
@with_kw struct FeedForwardNetwork <: NeuralNetwork
  # The type of the network
  type :: NetworkType

  # The state vector dimension at start of each layer
  xdims :: Vector{Int}
  @assert length(xdims) >= 3

  zdims :: Vector{Int} = [xdims[1:end-1]; 1]

  # Each M[K] == [Wk bk]
  Ms :: Vector{Matrix{Float64}}
  K :: Int = length(Ms)
  @assert length(xdims) == K + 1

  # Assert a non-trivial structural integrity of the network
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

# Generic input constraint supertype
abstract type InputConstraint end

# The set where {x : x1min <= x <= x1max}
@with_kw struct BoxInput <: InputConstraint
  x1min :: Vector{Float64}
  x1max :: Vector{Float64}
  @assert length(x1min) == length(x1max)
end

# The set where {x : Hx <= h}
@with_kw struct PolytopeInput <: InputConstraint
  H :: Matrix{Float64}
  h :: Vector{Float64}
  @assert size(H)[1] == length(h)
end

# Safety constraints
# The set {x : [x; f(x); 1]' * S * [x; f(x); 1] <= 0
@with_kw struct SafetyConstraint
  S :: Matrix{Float64}
end

# Reachability Constraints
abstract type ReachableSet end

# Given a hyperplane normals such that each normalk' * x <= hk
@with_kw struct HyperplaneSet <: ReachableSet
  normal :: Vector{Float64}
  @assert length(normal) >= 1
end

#
abstract type QueryInstance end

# A verification instance
@with_kw struct SafetyInstance <: QueryInstance
  ffnet :: FeedForwardNetwork
  input :: InputConstraint
  safety :: SafetyConstraint
end

# A hyerplane bounding instance
@with_kw struct ReachabilityInstance <: QueryInstance
  ffnet :: FeedForwardNetwork
  input :: InputConstraint
  reach_set :: ReachableSet
end

# The solution that is to be output by an algorithm
@with_kw struct SolutionOutput{A, B, C, D}
  objective_value :: A
  values :: B
  summary :: C
  termination_status :: D
  total_time :: Float64
  setup_time :: Float64
  solve_time :: Float64
end

#
export NetworkType, ReluNetwork, TanhNetwork
export NeuralNetwork, FeedForwardNetwork
export InputConstraint, BoxInput, PolytopeInput
export SafetyConstraint, ReachabilityConstraint, HyperplaneSet
export QueryInstance, SafetyInstance, ReachabilityInstance
export SolutionOutput

end # End module

