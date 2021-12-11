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
  nettype :: NetworkType

  # The state vector dimension at start of each layer
  xdims :: Vector{Int}
  @assert length(xdims) > 2

  zdims :: Vector{Int} = [xdims[1:end-1]; 1]

  # Each M[K] == [Wk bk]
  Ms :: Vector{Matrix{Float64}}
  K :: Int = length(Ms)
  @assert length(xdims) == K + 1

  # Assert a non-trivial structural integrity of the network
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

# Patterns that the T matirx may have
abstract type TPattern end
struct FullyDensePattern <: TPattern end
@with_kw struct BandedPattern <: TPattern
  tband :: Int
  @assert tband >= 0
end

# Generic input constraint supertype
abstract type InputConstraint end

# The set where {x : xbot <= x <= xtop}
@with_kw struct BoxConstraint <: InputConstraint
  xbot :: Vector{Float64}
  xtop :: Vector{Float64}
  @assert length(xbot) == length(xtop)
end

# The set where {x : Hx <= h}
@with_kw struct PolytopeConstraint <: InputConstraint
  H :: Matrix{Float64}
  h :: Vector{Float64}
  @assert size(H)[1] == length(h)
end

# Output constraints
abstract type OutputConstraint end

# The set {x : [x; f(x); 1]' * S * [x; f(x); 1] <= 0
@with_kw struct SafetyConstraint <: OutputConstraint
  S :: Matrix{Float64}
end

# Given a hyperplane normals c1, ..., cm, find d1, ..., dm such that each ck' * x <= dk
@with_kw struct HyperplanesConstraint <: OutputConstraint
  normals :: Vector{Vector{Float64}}
  @assert length(normals) >= 0

  # Sanity check, they must all be th same dimensions
  dims :: Vector{Int} = length.(normals)
  @assert all(y -> y == dims[1], dims)
end

#
abstract type QueryInstance end

# A verification instance
@with_kw struct SafetyInstance <: QueryInstance
  ffnet :: FeedForwardNetwork
  input :: InputConstraint
  safety :: SafetyConstraint

  # By default, no sparsity
  β :: Int = ffnet.K - 2
  @assert 1 <= β <= ffnet.K - 2

  pattern :: TPattern = FullyDensePattern()
  verbose :: Bool = false
end

# A hyerplane bounding instance
@with_kw struct ReachabilityInstance <: QueryInstance
  ffnet :: FeedForwardNetwork
  input :: InputConstraint
  hplanes :: HyperplanesConstraint

  # By default, no sparsity
  β :: Int = ffnet.K - 2
  @assert 1 <= β <= ffnet.K - 2

  pattern :: TPattern = FullyDensePattern()
  verbose :: Bool = false
end

# The solution that is to be output by an algorithm
@with_kw struct SolutionOutput{A, B, C}
  solution :: A
  summary :: B
  status :: C
  total_time :: Float64
  setup_time :: Float64
  solve_time :: Float64
end

#
export NetworkType, ReluNetwork, TanhNetwork
export NeuralNetwork, FeedForwardNetwork
export InputConstraint, BoxConstraint, PolytopeConstraint
export OutputConstraint, SafetyConstraint, HyperplanesConstraint
export TPattern, BandedPattern, FullyDensePattern
export QueryInstance, SafetyInstance, ReachabilityInstance
export SolutionOutput

end # End module

