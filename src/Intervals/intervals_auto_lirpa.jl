using PyCall

# Set up auto_lirpa_bridge stuff
AUTO_LIRPA_PARENT_DIR = joinpath(@__DIR__, "..", "..", "exts")
if !(AUTO_LIRPA_PARENT_DIR in PyVector(pyimport("sys")."path"))
  pushfirst!(PyVector(pyimport("sys")."path"), AUTO_LIRPA_PARENT_DIR)
end

auto_lirpa_bridge = pyimport("auto_lirpa_bridge")

# Slice up the ffnet
function sliceFeedFwdNet(ffnet::FeedFwdNet)
  slices = Vector{FeedFwdNet}()
  # For each 1 <= k <= K-1 the [Wk bk] is replaced by [Ik 0]
  # the kth slices bounds x[k+1]
  for k in 1:ffnet.K-1
    xdims = ffnet.xdims[1:k+1]
    xdims = [xdims; xdims[end]] # Repeat the last one because of identity
    Ms = ffnet.Ms[1:k]
    Ms = push!(Ms, [I(xdims[end]) zeros(xdims[end])]) # Append the [Ik 0]
    ffnetk = FeedFwdNet(activ=ffnet.activ, xdims=xdims, Ms=Ms)
    push!(slices, ffnetk)
  end

  # The last one is just the full network
  push!(slices, ffnet)
  return slices
end

# Calculate the bounds of a FeedFwdNet
function autoLirpaBoundsOutput(x1min::VecReal, x1max::VecReal, ffnet::FeedFwdNet)
  @assert length(x1min) == length(x1max)
  # Write the ffnet to an onnx file
  onnx_file = tempname()
  writeOnnx(ffnet, onnx_file)
  lb, ub = auto_lirpa_bridge.find_bounds_output(onnx_file, x1min, x1max)
  # We need to post-process this a little to account for numerical errors
  lb = min.(lb, ub)
  ub = max.(lb, ub)
  return lb, ub
end

# Interval propagation where we slice the network
function makeIntervalsInfo(::IntervalsAutoLirpaSliced, x1min::VecReal, x1max::VecReal, ffnet::FeedFwdNet)
  ffnets = sliceFeedFwdNet(ffnet)
  x_intvs = Vector{PairVecReal}()
  push!(x_intvs, (x1min, x1max))
  for ggnet in ffnets
    lb, ub = autoLirpaBoundsOutput(x1min, x1max, ggnet)
    @assert all(lb .<= ub)
    push!(x_intvs, (lb, ub))
  end

  acx_intvs = Vector{PairVecReal}()
  for k in 1:ffnet.K-1
    Wk, bk = ffnet.Ms[k][:, 1:end-1], ffnet.Ms[k][:, end]
    xkmin, xkmax = x_intvs[k]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk
    @assert all(ykmin .<= ykmax)
    push!(acx_intvs, (ykmin, ykmax))
  end
  return IntervalsInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs) 
end

# Interval propagation where we use internal nodes to get x_intvs at once
function makeIntervalsInfo(::IntervalsAutoLirpaOneShot, x1min::VecReal, x1max::VecReal, ffnet::FeedFwdNet)
  @assert length(x1min) == length(x1max)

  # Write the ffnet to an onnx file
  onnx_file = tempname()
  writeOnnx(ffnet, onnx_file)
  x_intvs, _ = auto_lirpa_bridge.find_bounds_one_shot(onnx_file, x1min, x1max)

  # Do the acx stuff
  acx_intvs = Vector{PairVecReal}()
  for k in 1:ffnet.K-1
    Wk, bk = ffnet.Ms[k][:, 1:end-1], ffnet.Ms[k][:, end]
    xkmin, xkmax = x_intvs[k]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk
    push!(acx_intvs, (ykmin, ykmax))
  end

  return IntervalsInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs)
end

