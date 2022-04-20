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
function findFeedFwdNetBounds(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64)
  @assert length(x1min) == length(x1max)
  if !(EXTS_DIR in PyVector(pyimport("sys")."path"))
    pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)
  end
end
