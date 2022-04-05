
# Make Q using bounds ϕlow and ϕhigh
@with_kw struct QcBoundedActiv <: QcActiv
  acxdim::Int
  acxmin::VecF64
  acxmax::VecF64
  @assert acxdim == length(acxmin) == length(acxmax)
  @assert all(acxmin .<= acxmax)
  vardim::Int = acxdim
end

# The construction of Qac to be used in Zac
function makeQac(γac, qc::QcBoundedActiv)
  @assert length(γac) == qc.vardim
  D = Diagonal(γac)
  _Q11 = spzeros(qc.acxdim, qc.acxdim)
  _Q12 = spzeros(qc.acxdim, qc.acxdim)
  _Q13 = spzeros(qc.acxdim)
  _Q22 = -2 * D
  _Q23 = D * (qc.acxmin + qc.acxmax)
  _Q33 = -2 * qc.acxmin' * D * qc.acxmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

