
# When we have information on the network output y
@with_kw struct QcActivFinal <: QcActiv
  ffnet::FeedFwdNet
  ymin::VecReal
  ymax::VecReal
  # @assert ffnet.xdims[end] == length(ymin) == length(ymax)
  # @assert all(ymin .<= ymax)
  vardim = length(ymin)
end

# The corresponding Q for when we can bound the network output
function makeQ(γac, qc::QcActivFinal)
  @assert length(γac) == qc.vardim
  ffnet = qc.ffnet
  xdims, K = ffnet.xdims, ffnet.K
  acdims, ydim = xdims[2:end-1], xdims[end]
  @assert length(acdims) == K-1
  acdim = sum(acdims)
  D = Diagonal(γac)

  # This is a constraint on the output
  _S11 = -2 * D
  _S12 = D * (qc.ymin + qc.ymax)
  _S22 = -2 * qc.ymin' * D * qc.ymax
  S = Symmetric([_S11 _S12; _S12' _S22])

  # To convert [xK; 1] into y, allowing us to do R' * S * R
  WK, bK = ffnet.Ms[K][:, 1:end-1], ffnet.Ms[K][:, end]
  _R11 = WK
  _R12 = bK
  _R21 = spzeros(1, size(WK, 2))
  _R22 = 1
  R = [_R11 _R12; _R21 _R22]

  # Selectors along [x2 ... xK], which comprise acdim
  # We need to project out [xK;1] from [W1x1 ... WK-1 xK-1; x2 ... xK; 1]
  Qdims = [acdims; acdims; 1]
  FxK = E(2*length(acdims), Qdims)
  Fa = E(2*length(acdims)+1, Qdims)
  F = [FxK; Fa]

  Q = F' * R' * S * R * F
  return Q
end

