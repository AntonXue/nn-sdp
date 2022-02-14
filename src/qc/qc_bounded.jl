using Parameters

using ..MyLinearAlgebra
using ..MyNeuralNetwork

# Make Q using bounds ϕlow and ϕhigh
@with_kw struct QcBoundedInfo <: QcInfo
  qxdim :: Int
  ϕmin :: VecF64
  ϕmax :: VecF64
  @assert qxdim == length(ϕmin) == length(ϕmax)
  @assert all(ϕmin .<= ϕmax)
end

# Calculate the size of the variable needed for this Qc
function vardim(qcinfo :: QcBoundedInfo)
  return qcinfo.qxdim
end

# The construction
function makeQc(γ, qcinfo :: QcBoundedInfo)
  qxdim, ϕmin, ϕmax = qcinfo.qxdim, qcinfo.ϕmin, qcinfo.ϕmax
  @assert length(γ) == vardim(qcinfo)

  D = diagm(γ) # TODO: more efficent encoding
  _Q11 = zeros(qxdim, qxdim)
  _Q12 = zeros(qxdim, qxdim)
  _Q13 = zeros(qxdim)
  _Q22 = -2 * D
  _Q23 = D * (ϕmin + ϕmax)
  _Q33 = -2 * ϕmin' * D * ϕmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end
