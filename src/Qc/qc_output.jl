
# Qc for safety
@with_kw struct QcSafety <: QcOutput
  S::MatF64
  @assert size(S) == size(S') # Is square
  vardim::Int = 0
end

# General reachability
abstract type QcReach <: QcOutput end

# Hyperplane reachability in particular
@with_kw struct QcHplaneReach <: QcReach
  normal::VecF64
  vardim::Int = 1
end

# With only a matrix
function makeSide(ffnet::FeedFwdNet)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  bK = ffnet.Ms[ffnet.K][1:end, end]
  d1 = ffnet.zdims[1]
  (dK1, dK) = size(WK)
  _R11 = I(d1)
  _R12 = spzeros(d1, dK)
  _R13 = spzeros(d1)
  _R21 = spzeros(dK1, d1)
  _R22 = WK
  _R23 = bK
  _R31 = spzeros(1, d1)
  _R32 = spzeros(1, dK)
  _R33 = 1
  R = [_R11 _R12 _R13; _R21 _R22 _R23; _R31 _R32 _R33]
  return R
end

# Safety Zout
function makeZout(qc::QcSafety, ffnet::FeedFwdNet)
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Eout = [E1; EK; Ea]
  R = makeSide(ffnet)
  Zout = Eout' * R' * qc.S * R * Eout
  return Zout
end

# Reach Zout
function makeZout(γout, qc::QcReach, ffnet::FeedFwdNet)
  @assert length(γout) == qc.vardim
  if qc isa QcHplaneReach
    d1 = ffnet.xdims[1]
    dK1 = ffnet.xdims[end]
    @assert length(qc.normal) == dK1
    _S11 = spzeros(d1, d1)
    _S12 = spzeros(d1, dK1)
    _S13 = spzeros(d1)
    _S22 = spzeros(dK1, dK1)
    _S23 = qc.normal
    _S33 = -2 * γout
    S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33] 
  else
    error("unrecognized qc: $(qc)")
  end

  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Eout = [E1; EK; Ea]
  R = makeSide(ffnet)
  Zout = Eout' * R' * S * R * Eout
  return Zout
end

