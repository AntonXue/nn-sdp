
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
  vardim::Int = lenght(h)^2
end

# Make different Zin depending on the QcInput
function makeZin(γin, qc::QcInput, ffnet::FeedFwdNet)
  @assert length(γin) == qc.vardim
  # Qc for boxes
  if qc isa QcInputBox
    Γ = Diagonal(γin)
    _P11 = -2 * Γ
    _P12 = Γ * (qc.x1min + qc.x1max)
    _P22 = -2 * qc.x1min' * Γ * qc.x1max
  # Qc for polytopes
  elseif qc isa QcInputPoly
    _P11 = H' * Γ * H
    _P12 = -H' * Γ * h
    _P22 = h' * Γ * h
  else
    error("unrecognized qc: $(qc)")
  end

  P = [_P11 _P12; _P12' _P22]
  E1 = E(1, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Zin = Ein' * P * Ein
  return Zin
end

