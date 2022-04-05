using ..MyNeuralNetwork

include("qc_activ_bounded.jl")
include("qc_activ_sector.jl")

# Make the A matrix
function makeA(ffnet::FeedFwdNet)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, fdims)' * Ws[k] * E(k, edims) for k in 1:(ffnet.K-1))
  return sparse(A)
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

# Make the Zac; call this once for each Qac used
function makeZac(γac, qc::QcActiv, ffnet::FeedFwdNet)
  @assert length(γac) == qc.vardim
  Qac = makeQac(γac, qc)
  _R11 = makeA(ffnet)
  _R12 = makeb(ffnet)
  _R21 = makeB(ffnet)
  _R22 = zeros(size(_R21)[1])
  _R31 = zeros(1, size(_R21)[2])
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  Zac = R' * Qac * R
  return Zac
end

