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

include("vnnlib_utils.jl")

# A general form of quadratic safety
# a||x||^2 + b||f(x)||^2 + c <= 0
function abcQuadS(a, b, c, ffnet::FeedFwdNet)
  xdims, K = ffnet.xdims, ffnet.K
  _S11 = a * I(xdims[1])
  _S12 = spzeros(xdims[1], xdims[K+1])
  _S13 = spzeros(xdims[1], 1)
  _S22 = b * I(xdims[K+1])
  _S23 = spzeros(xdims[K+1], 1)
  _S33 = c
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
end

# ||f(x)||^2 <= L ||x||^2
function L2S(L2gain, ffnet::FeedFwdNet)
  return abcQuadS(-L2gain, 1.0, 0.0, ffnet)
end

# ||f(x)||^2 <= C
function outNorm2S(norm2, ffnet::FeedFwdNet)
  return abcQuadS(0.0, 1.0, -norm2, ffnet)
end

# Hyperplane S
function hplaneS(normal, h, ffnet::FeedFwdNet)
  xdims, K = ffnet.xdims, ffnet.K
  _S11 = spzeros(xdims[1], xdims[1])
  _S12 = spzeros(xdims[1], xdims[K+1])
  _S13 = spzeros(xdims[1])
  _S22 = spzeros(xdims[K+1], xdims[K+1])
  _S23 = normal
  _S33 = -2 * h
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims::VecInt; activ::Activ = ReluActiv(), σ::Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = [randn(xdims[k+1], xdims[k]+1) * σ for k in 1:(length(xdims) - 1)]
  return FeedFwdNet(activ=activ, xdims=xdims, Ms=Ms)
end

# Generate trajectories from a unit box
function randomTrajectories(N::Int, ffnet::FeedFwdNet, x1min, x1max)
  # Random.seed!(1234) # Let's move the call to this earlier
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]
  xgaps = x1max - x1min
  box01points = rand(ffnet.xdims[1], N)
  x1s = [x1min + (p .* xgaps) for p in eachcol(box01points)]
  xfs = [evalFeedFwdNet(ffnet, x1) for x1 in x1s]
  return xfs
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

function plotBoundingPolys(points::Vector{VecF64}, labeled_polys::Vector{Tuple{String, Poly}}; saveto="~/dump/foo.png")
  @assert all(z -> z == 2, length.(points))
  @assert all(lbp -> length(lbp[2]) >= 3, labeled_polys)

  # The points
  xs = [p[1] for p in points]
  ys = [p[2] for p in points]

  # The vertices of each poly
  labeled_polyverts = [(label, polyVertices(poly)) for (label, poly) in labeled_polys]

  for (label, (vxs, vys)) in labeled_polyverts
    @printf("label: %s\n", label)
    println(round.([vxs vys], digits=2))
    println("")
  end

  # Figure out the extreme points to set the plot dimensions
  lpvxs = vcat([pvs[1] for (_, pvs) in labeled_polyverts]...)
  lpvys = vcat([pvs[2] for (_, pvs) in labeled_polyverts]...)

  xmin, xmax = minimum([xs; lpvxs]), maximum([xs; lpvxs])
  ymin, ymax = minimum([ys; lpvys]), maximum([ys; lpvys])
  xgap = xmax - xmin
  ygap = ymax - ymin
  plotxlim = (xmin - 0.3 * xgap, xmax + 0.3 * xgap)
  plotylim = (ymin - 0.3 * ygap, ymax + 0.3 * ygap)

  # Plot stuff
  cur_colors = theme_palette(:auto)
  plt = plot()
  for (i, (lbl, (vxs, vys))) in enumerate(labeled_polyverts)
    plt = plot!(vxs, vys, color=cur_colors[i], label=lbl)
  end

  plt = scatter!(xs, ys, markersize=4, alpha=0.3, color=:blue,
                xlim=plotxlim, ylim=plotylim, label="",
                legendfont=font(12),
                xtickfont=font(10),
                ytickfont=font(10))
  savefig(plt, saveto)
  return plt
end

# Convert NNet to FeedFwdNet

export abcQuadS, L2S, outNorm2S, hplaneS
export randomNetwork
export runNetwork, randomTrajectories, plotRandomTrajectories
export plotBoundingPolys

export loadVnnlib, loadReluQueries

end # End Module

