module Intervals

using LinearAlgebra
using Parameters
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork
using ..Files

# The result of interval propagation
@with_kw struct IntervalsInfo
  ffnet::FeedFwdNet

  # The ranges of each x
  x_intvs::Vector{PairVecF64}
  @assert length(x_intvs) == ffnet.K+1
  @assert all(xi -> length(xi) == 2, x_intvs)
  @assert all(xi -> length(xi[1]) == length(xi[2]), x_intvs)
  @assert all(k -> ffnet.xdims[k] == length(x_intvs[k][1]), 1:(ffnet.K+1))

  # What is fed into each activation
  acx_intvs::Vector{PairVecF64}
  @assert length(acx_intvs) == ffnet.K-1
  @assert all(acxi -> length(acxi) == 2, acx_intvs)
  @assert all(acxi -> length(acxi[1]) == length(acxi[2]), acx_intvs)
  @assert all(k -> ffnet.xdims[k+1] == length(acx_intvs[k][1]), 1:ffnet.K-1)
end

include("intervals_easy.jl")
include("intervals_auto_lirpa.jl")

export IntervalsInfo
export intervalsWorstCase, intervalsSampled, intervalsAutoLirpa

end

