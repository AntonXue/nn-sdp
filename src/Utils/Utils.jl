# Some useful utilities, such as file IO
# Should only be used within main.jl or a similar file
module Utils

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Random
using Plots
using Printf

using ..MyMath
using ..MyNeuralNetwork
using ..Qc
using ..Methods
using ..Intervals

pyplot()

include("qc.jl")
include("vnnlib.jl")
include("plots.jl")

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims::VecInt; activ::Activ = ReluActiv(), σ::Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = [randn(xdims[k+1], xdims[k]+1) * σ for k in 1:(length(xdims) - 1)]
  return FeedFwdNet(activ=activ, xdims=xdims, Ms=Ms)
end


# Plot some data to a file
function plotRandomTrajectories(N::Int, ffnet::FeedFwdNet, x1min, x1max; saveto="~/Desktop/hello.png")
  # Make sure we can actually plot these in 2D
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]
  @assert ffnet.xdims[end] == 2

  xfs = randomTrajectories(N, ffnet, x1min, x1max)
  d1s = [xf[1] for xf in xfs]
  d2s = [xf[2] for xf in xfs]
  
  p = scatter(d1s, d2s, markersize=2, alpha=0.3)
  savefig(p, saveto)
  return xfs
end

# Find the vertices of a bunch of hyperplanes
const Hplane = Tuple{VecF64, Float64}
const Poly = Vector{Hplane}

function polyVertices(poly::Poly)
  hplanes = poly
  augs = [hplanes; hplanes[1]; hplanes[2]]
  verts = Vector{VecF64}()
  for i in 1:(length(augs)-1)
    n1, h1 = augs[i]
    n2, h2 = augs[i+1]
    x = [n1'; n2'] \ [h1; h2]
    push!(verts, x)
  end
  vxs = [v[1] for v in verts]
  vys = [v[2] for v in verts]
  return vxs, vys
end

# Convert NNet to FeedFwdNet

export abcQuadS, L2S, outNorm2S, hplaneS
export randomNetwork
export runNetwork, randomTrajectories, plotRandomTrajectories
export plotBoundingPolys

export loadVnnlib, loadReluQueries

end # End Module

