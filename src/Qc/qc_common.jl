# Make the A matrix
function makeA(ffnet::FeedFwdNet)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, fdims)' * Ws[k] * E(k, edims) for k in 1:(ffnet.K-1))
  return A
end

# Make the b stacked vector
function makeb(ffnet::FeedFwdNet)
  bs = [M[1:end, end] for M in ffnet.Ms[1:end-1]]
  return vcat(bs...)
end

# Make the B matrix
function makeB(ffnet::FeedFwdNet)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:(ffnet.K-1))
  return B
end

