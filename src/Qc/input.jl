
# QC for box inputs
@with_kw struct QcInputBox <: QcInput
  x1min::VecReal
  x1max::VecReal
  @assert length(x1min) == length(x1max)
  vardim::Int = length(x1min)
end

# Qc for polytope inputs
@with_kw struct QcInputPoly <: QcInput
  H::MatReal
  h::VecReal
  @assert size(H)[1] == length(h)
  vardim::Int = length(h)^2
end

# P derived from box
function makeP(γin, qc::QcInputBox, ffnet::FeedFwdNet)
  Γ = Diagonal(γin)
  _P11 = -2 * Γ
  _P12 = Γ * (qc.x1min + qc.x1max)
  _P22 = -2 * qc.x1min' * Γ * qc.x1max
  P = [_P11 _P12; _P12' _P22]
  return P
end

# P derived from poly
function makeP(γin, qc::QcInputPoly, ffnet::FeedFwdNet)
  Γ = reshape(γin, length(qc), length(qc))
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

# The actual Zin
function makeZin(γin, qc::QcInput, ffnet::FeedFwdNet)
  @assert length(γin) == qc.vardim
  P = makeP(γin, qc, ffnet)
  E1 = E(1, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Zin = Ein' * P * Ein
  return Zin
end

