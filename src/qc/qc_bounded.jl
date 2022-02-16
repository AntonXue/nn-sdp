using Parameters
using ..MyLinearAlgebra
using ..MyNeuralNetwork

# Make Q using bounds ϕlow and ϕhigh
@with_kw struct QcBoundedInfo <: QcInfo
  qxdim :: Int
  acmin :: VecF64
  acmax :: VecF64
  @assert qxdim == length(acmin) == length(acmax)
  @assert all(acmin .<= acmax)
end

# Calculate the size of the variable needed for this Qc
function vardim(qcinfo :: QcBoundedInfo)
  return qcinfo.qxdim
end

# The construction
function makeQc(γ, qcinfo :: QcBoundedInfo)
  qxdim, acmin, acmax = qcinfo.qxdim, qcinfo.acmin, qcinfo.acmax
  @assert length(γ) == vardim(qcinfo)

  D = Diagonal(γ) # TODO: more efficent encoding
  _Q11 = zeros(qxdim, qxdim)
  _Q12 = zeros(qxdim, qxdim)
  _Q13 = zeros(qxdim)
  _Q22 = -2 * D
  _Q23 = D * (acmin + acmax)
  _Q33 = -2 * acmin' * D * acmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

