using Parameters
using SparseArrays

using ..MyLinearAlgebra
using ..MyNeuralNetwork

@with_kw struct QcSectorInfo <: QcInfo
  qxdim::Int
  β::Int
  smin::VecF64
  smax::VecF64
  base_smin::Float64
  base_smax::Float64
  @assert qxdim == length(smin) == length(smax)
  @assert 0 <= β
  @assert base_smin <= base_smax
  @assert smin <= smax
  @assert all(base_smin .<= smin)
  @assert all(smax .<= base_smax)
end

function vardim(qcinfo::QcSectorInfo)
  q, β = qcinfo.qxdim, qcinfo.β
  return sum((q-β):q)
end

function makeQc(γ, qcinfo::QcSectorInfo)
  @assert length(γ) == vardim(qcinfo)
  qxdim, β = qcinfo.qxdim, qcinfo.β
  initγ = γ[1:qxdim]

  if β > 0
    ijs = [(i, j) for i in 1:(qxdim-1) for j in (i+1):qxdim if j-i <= β]
    δts = [e(i, qxdim)' - e(j, qxdim)' for (i, j) in ijs]
    Δ = vcat(δts...)

    @assert qxdim + length(ijs) == length(γ)

    # Given a pair i,j, calculate its relative index in the γ vector
    v = vec([γ[qxdim+ind] for ind in 1:length(ijs)])
    T = Δ' * (v .* Δ)

    # pair2ind(i,j) = sum((qxdim-(j-i)+1):qxdim) + i
    # v = vec([γ[pair2ind(i,j)] for (i,j) in ijs])
    # T = Δ' * (v .* Δ)
  else
    T = spzeros(qxdim, qxdim)
  end

  base_smin, base_smax = qcinfo.base_smin, qcinfo.base_smax
  smin, smax = qcinfo.smin, qcinfo.smax
  _Q11 = -2 * Diagonal(smin .* smax.* initγ) - 2 * (base_smin * base_smax * T)
  _Q12 = Diagonal((smin + smax) .* initγ) + (base_smin + base_smax) * T
  _Q13 = spzeros(size(_Q11)[1])
  _Q22 = -2 * T
  _Q23 = spzeros(size(_Q22)[1])
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
end

function sectorBounds(qxmin::VecF64, qxmax::VecF64, activ::Activ)
  @assert length(qxmin) == length(qxmax)
  ε = 1e-4
  if activ isa ReluActiv
    Ipos = findall(z -> z > ε, qxmin)
    Ineg = findall(z -> z < -ε, qxmax)
    smin, smax = spzeros(length(qxmin)), ones(length(qxmax))
    smin[Ipos] .= 1.0
    smax[Ineg] .= 0.0
    return smin, smax

  elseif activ isa TanhActiv
    smin, smax = spzeros(length(qxmin)), ones(length(qxmax))
    for i in 1:length(qxmin)
      if qxmin[i] * qxmax[i] >= 0
        smin[i] = tanh(qxmax[i]) / qxmax[i]
        smax[i] = tanh(qxmin[i]) / qxmin[i]
      else
        smin[i] = min(tanh(qxmin[i]) / qxmin[i], tanh(qxmax[i]) / qxmax[i])
        smax[i] = 1.0
      end
    end
    return smin, smax

  else
    error("unsupported activation: $(activ)")
  end
end

