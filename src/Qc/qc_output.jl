
# Qc for safety
@with_kw struct QcSafety <: QcOutput
  S::Union{MatF64, SpMatF64}
  @assert size(S) == size(S') # Is square
  vardim::Int = 0
end

# General reachability
abstract type QcReach <: QcOutput end

# Hyperplane reachability in particular
@with_kw struct QcReachHplane <: QcReach
  normal::VecF64
  vardim::Int = 1
end

# Reachability of form ||y - y0||^2 <= ρ, where ρ = r^2
@with_kw struct QcReachCircle <: QcReach
  y0::VecF64
  vardim::Int = 1
end

# The R to be used in R' * S * R
function makeSide(ffnet::FeedFwdNet)
  xdims, K = ffnet.xdims, ffnet.K
  WK = ffnet.Ms[K][1:end, 1:end-1]
  bK = ffnet.Ms[K][1:end, end]
  _R11 = I(xdims[1])
  _R12 = spzeros(xdims[1], xdims[K])
  _R13 = spzeros(xdims[1])
  _R21 = spzeros(xdims[K+1], xdims[1])
  _R22 = WK
  _R23 = bK
  _R31 = spzeros(1, xdims[1])
  _R32 = spzeros(1, xdims[K])
  _R33 = 1
  R = [_R11 _R12 _R13; _R21 _R22 _R23; _R31 _R32 _R33]
  return R
end

# Safety Zout
function makeZout(qc::QcSafety, ffnet::FeedFwdNet)
  zdims, K = ffnet.zdims, ffnet.K
  E1 = E(1, zdims)
  EK = E(K, zdims)
  Ea = E(K+1, zdims)
  Eout = [E1; EK; Ea]
  R = makeSide(ffnet)
  Zout = Eout' * R' * qc.S * R * Eout
  return Zout
end

# Reach Zout
function makeZout(γout, qc::QcReach, ffnet::FeedFwdNet)
  @assert length(γout) == qc.vardim
  xdims, zdims, K = ffnet.xdims, ffnet.zdims, ffnet.K
  if qc isa QcReachHplane
    @assert length(qc.normal) == xdims[K+1]
    @assert qc.vardim == length(γout) == 1
    _S11 = spzeros(xdims[1], xdims[1])
    _S12 = spzeros(xdims[1], xdims[K+1])
    _S13 = spzeros(xdims[1])
    _S22 = spzeros(xdims[K+1], xdims[K+1])
    _S23 = qc.normal
    _S33 = -2 * γout
    S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  elseif qc isa QcReachCircle
    @assert length(qc.y0) == xdims[K+1]
    @assert qc.vardim == length(γout) == 1
    _S11 = spzeros(xdims[1], xdims[1])
    _S12 = spzeros(xdims[1], xdims[K+1])
    _S13 = spzeros(xdims[1])
    _S22 = I(xdims[K+1])
    _S23 = -qc.y0
    _S33 = qc.y0' * qc.y0 - γout[1]
    S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  else
    error("unrecognized qc: $(qc)")
  end

  E1 = E(1, zdims)
  EK = E(K, zdims)
  Ea = E(K+1, zdims)
  Eout = [E1; EK; Ea]
  R = makeSide(ffnet)
  Zout = Eout' * R' * S * R * Eout
  return Zout
end

# Apply scaling to S
function scaleS(S, αs::VecF64, ffnet::FeedFwdNet)
  @assert length(αs) == ffnet.K
  α = prod(αs)
  sdims = [ffnet.xdims[1]; ffnet.xdims[end]; 1]
  F1 = E(1, sdims)
  F2 = E(2, sdims)
  F3 = E(3, sdims)
  _S11 = F1 * S * F1' * α^2
  _S12 = F1 * S * F2' * α
  _S13 = F1 * S * F3' * α
  _S22 = F2 * S * F2'
  _S23 = F2 * S * F3'
  _S33 = F3 * S * F3'
  newS = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return newS
end


