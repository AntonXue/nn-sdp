
# QC for box inputs
@with_kw struct QcInputBox <: QcInput
  x1min::VecReal
  x1max::VecReal
  @assert length(x1min) == length(x1max)
  vardim::Int = length(x1min)
end

# Unlike QcSafety with S, there is no explicit P, so treat scaled input Qcs as their own thing

# Input box but with scaling
@with_kw struct QcInputBoxScaled <: QcInput
  x1min::VecReal
  x1max::VecReal
  α::Real
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

# Qc for polytope inputs but with scaling
@with_kw struct QcInputPolyScaled <: QcInput
  H::MatReal
  h::VecReal
  α::Real
  @assert size(H)[1] == length(h)
  vardim::Int = lenght(h)^2
end

# Make different Zin depending on the QcInput
function makeZin(γin, qc::QcInput, ffnet::FeedFwdNet)
  @assert length(γin) == qc.vardim
  # Qc for boxes
  if qc isa QcInputBox || qc isa QcInputBoxScaled
    Γ = Diagonal(γin)
    _P11 = -2 * Γ
    _P12 = Γ * (qc.x1min + qc.x1max)
    _P22 = -2 * qc.x1min' * Γ * qc.x1max

    if qc isa QcInputBoxScaled
      _P11 *= qc.α^2
      _P12 *= qc.α
    end
    P = [_P11 _P12; _P12' _P22]

  # Qc for polytopes
  elseif qc isa QcInputPoly || qc isa QcInputPolyScaled
    _P11 = H' * Γ * H
    _P12 = -H' * Γ * h
    _P22 = h' * Γ * h

    if qc isa QcInputPolyScaled
      _P11 *= qc.α^2
      _P12 *= qc.α
    end
    P = [_P11 _P12; _P12' _P22]
  else
    error("unrecognized qc: $(qc)")
  end

  # Put this together in a Zin
  E1 = E(1, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Zin = Ein' * P * Ein
  return Zin
end

