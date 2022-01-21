# Some useful utilities, such as file IO
# Should only be used within main.jl or a similar file
module Utils

using ..Header
using ..Common
using ..NNetParser: NNet
using LinearAlgebra
using DelimitedFiles
using Random
using Plots
using Printf

pyplot()

# A general form of quadratic safety
# a||x||^2 + b||f(x)||^2 + c <= 0
function quadraticSafety(a, b, c, xdims :: VecInt)
  @assert length(xdims) > 1
  _S11 = a * I(xdims[1])
  _S12 = zeros(xdims[1], xdims[end])
  _S13 = zeros(xdims[1], 1)
  _S22 = b * I(xdims[end])
  _S23 = zeros(xdims[end], 1)
  _S33 = c
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return SafetyConstraint(S=S)
end

# ||f(x)||^2 <= L ||x||^2
function L2gainSafety(L2gain, xdims :: VecInt)
  return quadraticSafety(-L2gain, 1.0, 0.0, xdims)
end

# ||f(x)||^2 <= C
function outputNorm2Safety(norm2, xdims :: VecInt)
  return quadraticSafety(0.0, 1.0, -norm2)
end

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims :: VecInt; type :: NetworkType = ReluNetwork(), σ :: Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = Vector{Any}()
  for k = 1:length(xdims) - 1
    # Width is xdims[k]+1 because Mk = [Wk bk]
    Mk = randn(xdims[k+1], xdims[k]+1) * σ
    push!(Ms, Mk)
  end
  return FeedForwardNetwork(type=type, xdims=xdims, Ms=Ms)
end

# Run a feedforward net on an initial input and give the output
function runNetwork(x1 :: VecF64, ffnet :: FeedForwardNetwork)
  @assert length(x1) == ffnet.xdims[1]
  function ϕ(x)
    if ffnet.type isa ReluNetwork; return max.(x, 0)
    elseif ffnet.type isa TanhNetwork; return tanh.(x)
    else; error("unsupported network: " * string(ffnet))
    end
  end

  xk = x1
  # Run through each layer
  for Mk in ffnet.Ms[1:end-1]
    xk = Mk * [xk; 1]
    xk = ϕ(xk)
  end
  # Then the final layer does not have an activation
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

# Generate trajectories from a unit box
function randomTrajectories(N :: Int, ffnet :: FeedForwardNetwork, x1min, x1max)
  # Random.seed!(1234) # Let's move the call to this earlier
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]
  xgaps = x1max - x1min
  box01points = rand(ffnet.xdims[1], N)
  x1s = [x1min + (p .* xgaps) for p in eachcol(box01points)]
  xfs = [runNetwork(x1, ffnet) for x1 in x1s]
  return xfs
end

# Plot some data to a file
function plotRandomTrajectories(N :: Int, ffnet :: FeedForwardNetwork, x1min, x1max; saveto="~/Desktop/hello.png")
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
const Polytope = Vector{Hplane}

function polytopeVertices(polytope :: Polytope)
  hplanes = polytope
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

function plotBoundingPolytopes(points :: Vector{VecF64}, labeled_polys :: Vector{Tuple{String, Polytope}}; saveto="~/Desktop/foo.png")
  @assert all(z -> z == 2, length.(points))
  @assert all(lbp -> length(lbp[2]) >= 3, labeled_polys)

  # The points
  xs = [p[1] for p in points]
  ys = [p[2] for p in points]

  # The vertices of each poly
  labeled_polyverts = [(label, polytopeVertices(poly)) for (label, poly) in labeled_polys]

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

# Convert NNet to FeedForwardNetwork + BoxInput
function NNet2FeedForwardNetwork(nnet :: NNet)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  ffnet = FeedForwardNetwork(type=ReluNetwork(), xdims=nnet.layerSizes, Ms=Ms)
  return ffnet
end

#
export quadraticSafety, L2gainSafety, outputNorm2Safety
export randomNetwork
export runNetwork, randomTrajectories, plotRandomTrajectories
export plotBoundingPolytopes
export NNet2FeedForwardNetwork

end # End Module

