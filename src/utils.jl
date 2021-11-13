# Some useful utilities, such as file IO
module Utils

using ..Header
using LinearAlgebra
using DelimitedFiles
using Random

# Write some data (hopefully Float64-based) to a file
function fileWriteFloat64(data, file :: String)
  open(file, "w") do io
    writedlm(io, data, ',')
  end
end

# Read some Float64-based data from a file
function fileReadFloat64(file :: String)
  data = readdlm(file, ',', Float64)
  return data
end

# Generate a random network given the desired dimensions at each layer
function randomReluNetwork(xdims :: Vector{Int64})
  @assert length(xdims) > 1
  M = Vector{Any}()
  for k = 1:length(xdims) - 1
    # Width is xsimds[k]+1 because Mk = [Wk bk]
    Mk = randn(xdims[k+1], xdims[k]+1)
    push!(M, Mk)
  end
  return FeedForwardNetwork(nettype=ReluNetwork(), xdims=xdims, M=M)
end

# -ones <= x <= ones
function inputUnitBox(xdims :: Vector{Int64})
  @assert length(xdims) > 1
  xbot = -ones(xdims[1])
  xtop = ones(xdims[1])
  return BoxConstraint(xbot=xbot, xtop=xtop)
end

# ||f(x)||^2 <= C
function safetyNormBound(C, xdims :: Vector{Int64})
  @assert C > 0
  @assert length(xdims) > 1
  _S11 = zeros(xdims[1], xdims[1])
  _S12 = zeros(xdims[1], xdims[end])
  _S13 = zeros(xdims[1], 1)
  _S22 = I(xdims[end])
  _S23 = zeros(xdims[end], 1)
  _S33 = -C
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return SafetyConstraint(S=S)
end


export fileWriteFloat64, fileReadFloat64
export randomReluNetwork
export inputUnitBox, safetyNormBound

end # End Module

