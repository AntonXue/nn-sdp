module MyNeuralNetwork

using Parameters
using ..MyLinearAlgebra

# The type of neural activation
abstract type Activation end
struct ReluActivation <: Activation end
struct TanhActivation <: Activation end

# Parameters needed to define the feed-forward network
@with_kw struct NeuralNetwork
  activ :: Activation

  # The state vector dimension at start of each layer
  xdims :: VecInt
  zdims :: VecInt = [xdims[1:end-1]; 1]
  @assert length(xdims) >= 3

  # Each M[K] == [Wk bk]
  Ms :: Vector{MatF64}
  K :: Int = length(Ms)
  @assert length(xdims) == K + 1

  # Assert a non-trivial structural integrity of the network
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

export Activation, ReluActivation, TanhActivation
export NeuralNetwork

end
