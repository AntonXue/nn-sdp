
# Make Q using bounds ϕlow and ϕhigh
@with_kw struct QcActivBounded <: QcActiv
  acydim::Int
  acymin::VecF64
  acymax::VecF64
  @assert acydim == length(acymin) == length(acymax)
  @assert all(acymin .<= acymax)
  vardim::Int = acydim
end

# The construction of Qac to be used in Zac
function makeQac(γac, qc::QcActivBounded)
  @assert length(γac) == qc.vardim
  D = Diagonal(γac)
  _Q11 = spzeros(qc.acydim, qc.acydim)
  _Q12 = spzeros(qc.acydim, qc.acydim)
  _Q13 = spzeros(qc.acydim)
  _Q22 = -2 * D
  _Q23 = D * (qc.acymin + qc.acymax)
  _Q33 = -2 * qc.acymin' * D * qc.acymax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

