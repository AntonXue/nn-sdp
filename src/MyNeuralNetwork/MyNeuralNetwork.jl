module MyNeuralNetwork

using Parameters
using ..MyMath

# The type of neural activation
abstract type Activ end
struct ReluActiv <: Activ end
struct TanhActiv <: Activ end

makeActiv(::ReluActiv) = return x -> max.(x, 0)
makeActiv(::TanhActiv) = return x -> tanh.(x)
makeActiv(::Activ) = error("not implemented")

activString(::ReluActiv) = "Relu"
activString(::TanhActiv) = "Tanh"

# Parameters needed to define the feed-forward network
@with_kw struct FeedFwdNet
  activ::Activ
  activ_func::Function = makeActiv(activ)

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

# Evaluate the network
function run(ffnet::FeedFwdNet, x)
  @assert length(x) == ffnet.xdims[1]
  xk = x
  for Mk in ffnet.Ms[1:end-1]; xk = ffnet.activ_func(Mk * [xk; 1]) end
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

export FeedFwdNet, makeActiv, run

include("network_files.jl")
export Activ, ReluActiv, TanhActiv
export makeActiv, activString
export load, loadScaled
export onnx2nnet, nnet2onnx, writeNnet, writeOnnx
export ScalingMode, NoScaling, SqrtLogScaling, FixedNormScaling, FixedConstScaling

end

