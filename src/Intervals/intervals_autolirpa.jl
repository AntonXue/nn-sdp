# Propagation with autolirpa

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

# Write ffnet to .nnet format, and save the result to a tmp file
function writeFeedFwdNet(ffnet::FeedFwdNet, saveto="$(homedir())/dump/hello.nnet")
  nnet_file = tempname()
  open(nnet_file, "w") do f
  end
end


