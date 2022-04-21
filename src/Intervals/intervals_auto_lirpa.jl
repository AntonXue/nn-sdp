using PyCall

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
function autoLirpaOutBounds(x1min::VecF64, x1max::VecF64, ffnet::FeedFwdNet)
  @assert length(x1min) == length(x1max)
  if !(EXTS_DIR in PyVector(pyimport("sys")."path"))
    pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)
  end

  # Write the ffnet to an onnx file
  onnx_file = tempname()
  writeOnnx(ffnet, onnx_file)
  bridge = pyimport("auto_lirpa_bridge")
  lb, ub = bridge.find_bounds(onnx_file, x1min, x1max)
  return lb, ub
end

# Interval propagation with auto_LiRPA
function intervalsAutoLirpa(x1min::VecF64, x1max::VecF64, ffnet::FeedFwdNet)
  ffnets = sliceFeedFwdNet(ffnet)
  x_intvs = Vector{PairVecF64}()
  push!(x_intvs, (x1min, x1max))
  for ggnet in ffnets
    lb, ub = autoLirpaOutBounds(x1min, x1max, ggnet)
    push!(x_intvs, (lb, ub))
  end

  acx_intvs = Vector{PairVecF64}()
  for k in 1:ffnet.K-1
    Wk, bk = ffnet.Ms[k][:, 1:end-1], ffnet.Ms[k][:, end]
    xkmin, xkmax = x_intvs[k]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk
    push!(acx_intvs, (ykmin, ykmax))
  end

  return IntervalsInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs)
end
