module Intervals

using LinearAlgebra
using Parameters
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork

# The result of interval propagation
@with_kw struct IntervalInfo
  nnet::NeuralNetwork

  # The ranges of each x
  x_intvs::Vector{PairVecF64}
  @assert length(x_intvs) == nnet.K+1
  @assert all(xi -> length(xi) == 2, x_intvs)
  @assert all(xi -> length(xi[1]) == length(xi[2]), x_intvs)
  @assert all(k -> nnet.xdims[k] == length(x_intvs[k][1]), 1:(nnet.K+1))

  # What is fed into each activation
  qx_intvs :: Vector{PairVecF64}
  @assert length(qx_intvs) == nnet.K-1
  @assert all(ϕi -> length(ϕi) == 2, qx_intvs)
  @assert all(ϕi -> length(ϕi[1]) == length(ϕi[2]), qx_intvs)
  @assert all(k -> nnet.xdims[k+1] == length(qx_intvs[k][1]), 1:nnet.K-1)
end

include("intervals_easy.jl")

export IntervalsInfo
export intervalsWorstCase

end
