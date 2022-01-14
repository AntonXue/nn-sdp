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

pyplot()

# a||x||^2 + b||f(x)||^2 <= C
function outputSafetyNorm2(a :: Float64, b :: Float64, norm2, xdims :: Vector{Int64})
  @assert length(xdims) > 1
  _S11 = a * I(xdims[1])
  _S12 = zeros(xdims[1], xdims[end])
  _S13 = zeros(xdims[1], 1)
  _S22 = b * I(xdims[end])
  _S23 = zeros(xdims[end], 1)
  _S33 = -norm2
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return SafetyConstraint(S=S)
end

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims :: Vector{Int64}; type :: NetworkType = ReluNetwork(), σ :: Float64 = 1.0)
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
function runNetwork(x1 :: Vector{Float64}, ffnet :: FeedForwardNetwork)
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

# Plot bouding hyperplanes for 2D points
function plotReachPolytope(points :: Vector{Vector{Float64}}, hplanes :: Vector{Tuple{Vector{Float64}, Float64}}; saveto="~/Desktop/foo.png")
  @assert all(z -> z == 2, length.(points))
  @assert length(hplanes) >= 3
  @assert all(hp -> length(hp[1]) == 2 && hp[2] isa Float64, hplanes)

  xs = [p[1] for p in points]
  ys = [p[2] for p in points]

  augs = [hplanes; hplanes[1]; hplanes[2]]

  verts = Vector{Any}()
  for i in 1:(length(augs)-1)
    n1, h1 = augs[i]
    n2, h2 = augs[i+1]
    x = [n1'; n2'] \ [h1; h2]
    # println("pushed: " * string(x))
    push!(verts, x)
  end

  vxs = [v[1] for v in verts]
  vys = [v[2] for v in verts]

  # Figure out how to plot
  xmin, xmax = minimum([xs; vxs]), maximum([xs; vxs])
  ymin, ymax = minimum([ys; vys]), maximum([ys; vys])
  xgap = xmax - xmin
  ygap = ymax - ymin
  plotxlim = (xmin - 0.4 * xgap, xmax + 0.4 * xgap)
  plotylim = (ymin - 0.4 * ygap, ymax + 0.4 * ygap)

  # Begin plotting
  plt = plot()
  plt = plot!(vxs, vys, color=:red)
  plt = scatter!(xs, ys, markersize=4, alpha=0.3, color=:blue, xlim=plotxlim, ylim=plotylim)
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
export outputSafetyNorm2
export randomNetwork
export runNetwork, randomTrajectories, plotRandomTrajectories
export plotReachPolytope
export NNet2FeedForwardNetwork

end # End Module

