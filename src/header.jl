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

  # Each M[K] == [Wk bk]
  M :: Vector{Matrix{Float64}}
  K :: Int = length(M)

  # Assert a non-trivial structural integrity of the network
  @assert length(xdims) > 2
  @assert length(xdims) == K + 1
  @assert [size(Mk) for Mk in M] == [(xdims[k+1], xdims[k] + 1) for k in 1:K]
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

# The set {x : [x; f(x); 1]' * S * [x; f(x); 1] <= 0
@with_kw struct SafetyConstraint
  S :: Matrix{Float64}
end

# The solution that is to be output by an algorithm
@with_kw struct SolutionOutput{M, S}
  model :: M
  summary :: S
  status :: String
  total_time :: Float64
  solve_time :: Float64
end

export NetworkType, ReluNetwork, TanhNetwork
export NeuralNetwork, FeedForwardNetwork
export InputConstraint, BoxConstraint, PolytopeConstraint
export SafetyConstraint
export SolutionOutput

end # End module

