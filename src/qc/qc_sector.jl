using Parameters

using ..MyLinearAlgebra
using ..MyNeuralNetwork

@with_kw struct QcSectorInfo <: QcInfo
  qxdim :: Int
  tband :: Int
  pre_a :: VecF64
  pre_b :: VecF64
  base_a :: Float64
  base_b :: Float64
  @assert qxdim == length(pre_a) == length(pre_b)
  @assert 0 <= tband
  @assert base_a <= base_b
  @assert pre_a <= pre_b
  @assert all(base_a .<= pre_a)
  @assert all(pre_b .<= base_b)
end

function vardim(qcinfo :: QcSectorInfo)
  q, t = qcinfo.qxdim, qcinfo.tband
  return sum((q-t):q)
end

function makeQc(γ, qcinfo :: QcSectorInfo)
  @assert length(γ) == vardim(qcinfo)

  qxdim, tband = qcinfo.qxdim, qcinfo.tband
  initγ = γ[1:qxdim]

  if tband > 0
    ijs = [(i, j) for i in 1:(qxdim-1) for j in (i+1):qxdim if j-i <= tband]
    δts = [e(i, qxdim)' - e(j, qxdim)' for (i, j) in ijs]
    Δ = vcat(δts...)

    # Given a pair i,j, calculate its relative index in the γ vector
    pair2ind(i,j) = sum((qxdim-(j-i)+1):qxdim) + i
    v = vec([γ[pair2ind(i,j)] for (i,j) in ijs])
    T = Δ' * (v .* Δ)
  else
    T = zeros(qxdim, qxdim)
  end

  base_a, base_b = qcinfo.base_a, qcinfo.base_b
  pre_a, pre_b = qcinfo.pre_a, qcinfo.pre_b
  _Q11 = -2 * Diagonal(pre_a .* pre_b.* initγ) - 2 * (base_a * base_b * T)
  _Q12 = Diagonal((pre_a + pre_b) .* initγ) + (base_a + base_b) * T
  _Q13 = zeros(size(_Q11)[1])
  _Q22 = -2 * T
  _Q23 = zeros(size(_Q22)[1])
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
end

function sectorBounds(pre_acmin :: VecF64, pre_acmax :: VecF64, activ :: Activation)
  @assert length(pre_acmin) == length(pre_acmax)
  ε = 1e-4
  if activ isa ReluActivation
    Ipos = findall(z -> z > ε, pre_acmin)
    Ineg = findall(z -> z < -ε, pre_acmax)
    pre_a, pre_b = zeros(length(pre_acmin)), ones(length(pre_acmax))
    pre_a[Ipos] .= 1.0
    pre_b[Ineg] .= 0.0
    return pre_a, pre_b

  elseif activ isa TanhActivation
    pre_a, pre_b = zeros(length(pre_ac_min)), ones(length(pre_ac_max))
    for i in 1:length(pre_ac_min)
      if pre_ac_min[i] * pre_ac_max[i] >= 0
        pre_a[i] = tanh(pre_ac_max[i]) / pre_ac_max[i]
        pre_b[i] = tanh(pre_ac_min[i]) / pre_ac_min[i]
      else
        pre_a[i] = min(tanh(pre_acmin[i]) / pre_acmin[i], tanh(pre_acmax[i]) / pre_acmax[i])
        pre_b[i] = 1.0
      end
    end
    return pre_a, pre_b

  else
    error(@sprintf("unsupported activation: %s\n", activ))
  end
end

