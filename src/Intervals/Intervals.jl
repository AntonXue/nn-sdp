module Intervals

using LinearAlgebra
using Parameters

using ..MyMath
using ..MyNeuralNetwork

# Different interval propagation methods
abstract type IntervalsMethod end
struct IntervalsWorstCase <: IntervalsMethod end
struct IntervalsSampled <: IntervalsMethod end
struct IntervalsAutoLirpa <: IntervalsMethod end

# The result of interval propagation
@with_kw struct IntervalsInfo
  ffnet::FeedFwdNet

  # The ranges of each x
  x_intvs::Vector{PairVecReal}
  @assert length(x_intvs) == ffnet.K+1
  @assert all(xi -> length(xi) == 2, x_intvs)
  @assert all(xi -> length(xi[1]) == length(xi[2]), x_intvs)
  @assert all(k -> ffnet.xdims[k] == length(x_intvs[k][1]), 1:(ffnet.K+1))

  # What is fed into each activation
  acx_intvs::Vector{PairVecReal}
  @assert length(acx_intvs) == ffnet.K-1
  @assert all(acxi -> length(acxi) == 2, acx_intvs)
  @assert all(acxi -> length(acxi[1]) == length(acxi[2]), acx_intvs)
  @assert all(k -> ffnet.xdims[k+1] == length(acx_intvs[k][1]), 1:ffnet.K-1)
end

include("intervals_easy.jl")
include("intervals_auto_lirpa.jl")

# Do interval calculations based on the method
function makeIntervalsInfo(x1min::VecReal, x1max::VecReal, ffnet::FeedFwdNet, method::IntervalsMethod = IntervalsAutoLirpa())
  if method isa IntervalsWorstCase
    return intervalsWorstCase(x1min, x1max, ffnet)
  elseif method isa IntervalsSampled
    return intervalsSampled(x1min, x1max, ffnet)
  elseif method isa IntervalsAutoLirpa
    return intervalsAutoLirpaSliced(x1min, x1max, ffnet)
  else
    error("unrecognized method: $(method)")
  end
end

export IntervalsMethod, IntervalsWorstCase, IntervalsSampled, IntervalsAutoLirpa
export IntervalsInfo
export makeIntervalsInfo

end

