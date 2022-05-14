module MyNeuralNetwork

using Parameters
using ..MyMath

# The type of neural activation
abstract type Activ end
struct ReluActiv <: Activ end
struct TanhActiv <: Activ end

# Parameters needed to define the feed-forward network
@with_kw struct FeedFwdNet
  activ::Activ

  # The state vector dimension at start of each layer
  xdims::VecInt
  zdims::VecInt = [xdims[1:end-1]; 1]
  @assert length(xdims) >= 3

  # Each M[K] == [Wk bk]
  Ms::Vector{MatReal}
  K::Int = length(Ms)
  @assert length(xdims) == K + 1

  # Assert a non-trivial structural integrity of the network
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

function makeActiv(activ::Activ)
  if activ isa ReluActiv
    return x -> max.(x, 0)
  elseif activ isa TanhActiv
    return x -> tanh.(x)
  else
    error("unrecognized activ: $(activ)")
  end
end

# Evaluate the network
function evalFeedFwdNet(ffnet::FeedFwdNet, x)
  @assert length(x) == ffnet.xdims[1]
  xk, ac = x, makeActiv(ffnet.activ)
  for Mk in ffnet.Ms[1:end-1]; xk = ac(Mk * [xk; 1]) end
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

export Activ, ReluActiv, TanhActiv
export FeedFwdNet, makeActiv, evalFeedFwdNet

include("network_files.jl")
export loadFromNnet, loadFromOnnx, loadFromFile
export onnx2nnet, nnet2onnx, writeNnet, writeOnnx

export ScalingMode, NoScaling, SqrtLogScaling, FixedNormScaling, FixedConstScaling
export loadFromFileScaled

end

