using Parameters
using SparseArrays
using ..MyLinearAlgebra
using ..MyNeuralNetwork

# Make Q using bounds ϕlow and ϕhigh
@with_kw struct QcBoundedInfo <: QcInfo
  qxdim::Int
  qxmin::VecF64
  qxmax::VecF64
  @assert qxdim == length(qxmin) == length(qxmax)
  @assert all(qxmin .<= qxmax)
end

# Calculate the size of the variable needed for this Qc
function vardim(qcinfo::QcBoundedInfo)
  return qcinfo.qxdim
end

# The construction
function makeQc(γ, qcinfo::QcBoundedInfo)
  qxdim, qxmin, qxmax = qcinfo.qxdim, qcinfo.qxmin, qcinfo.qxmax
  @assert length(γ) == vardim(qcinfo)

  D = Diagonal(γ) # TODO: more efficent encoding
  _Q11 = spzeros(qxdim, qxdim)
  _Q12 = spzeros(qxdim, qxdim)
  _Q13 = spzeros(qxdim)
  _Q22 = -2 * D
  _Q23 = D * (qxmin + qxmax)
  _Q33 = -2 * qxmin' * D * qxmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

