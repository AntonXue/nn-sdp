
@with_kw struct QcActivSector <: QcActiv
  activ::Activ
  acxdim::Int
  β::Int
  base_smin::Real
  base_smax::Real
  # Default values; only need to specify the variables above this line
  smin::VecReal = ones(acxdim) * base_smin
  smax::VecReal = ones(acxdim) * base_smax
  @assert acxdim == length(smin) == length(smax)
  @assert 0 <= β
  @assert base_smin <= base_smax
  @assert smin <= smax
  @assert all(base_smin .<= smin)
  @assert all(smax .<= base_smax)
  # vardim::Int = sum((acxdim-β):acxdim)
  _λdim::Int = sum((acxdim-β):acxdim) # The base amount
  vardim::Int = (activ isa ReluActiv) ? _λdim + 2 * acxdim : _λdim
end

# The construction of Q, which will be used in Zac
function makeQ(γac, qc::QcActivSector)
  @assert length(γac) == qc.vardim
  acxdim, β = qc.acxdim, qc.β
  initγac = γac[1:acxdim]

  if β > 0
    ijs = [(i, j) for i in 1:(acxdim-1) for j in (i+1):acxdim if j-i <= β]
    δts = [e(i, acxdim)' - e(j, acxdim)' for (i, j) in ijs]
    Δ = vcat(δts...)
    @assert acxdim + length(ijs) == qc._λdim
    # Given a pair i,j, calculate its relative index in the γac vector
    v = vec([γac[acxdim+ind] for ind in 1:length(ijs)])
    T = Δ' * (v .* Δ)
  else
    T = spzeros(acxdim, acxdim)
  end

  base_smin, base_smax = qc.base_smin, qc.base_smax
  smin, smax = qc.smin, qc.smax
  _Q11 = -2 * Diagonal(smin .* smax.* initγac) - 2 * (base_smin * base_smax * T)
  _Q12 = Diagonal((smin + smax) .* initγac) + (base_smin + base_smax) * T
  _Q13 = spzeros(size(_Q11)[1])
  _Q22 = -2 * T
  _Q23 = spzeros(size(_Q22)[1])
  _Q33 = 0

  if qc.activ isa ReluActiv
    λend = qc._λdim
    ηstart, ηend = (qc._λdim + 1), (qc._λdim + qc.acxdim)
    νstart, νend = ηend + 1, qc.vardim
    η = γac[ηstart:ηend]
    ν = γac[νstart:νend]
    _Q13 = -smin .* η - smax .* ν
    _Q23 = η + ν
  end

  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
end

# relu sector smin / smax
function makeSectorMinMax(acxmin::VecReal, acxmax::VecReal, ::ReluActiv)
  @assert length(acxmin) == length(acxmax)
  ε = 1e-4
  Ipos = findall(z -> z > ε, acxmin)
  Ineg = findall(z -> z < -ε, acxmax)
  smin, smax = zeros(length(acxmin)), ones(length(acxmax))
  smin[Ipos] .= 1.0
  smax[Ineg] .= 0.0
  return smin, smax
end

# tanh sector smin / smax
function makeSectorMinMax(acxmin::VecReal, acxmax::VecReal, ::TanhActiv)
  @assert length(acxmin) == length(acxmax)
  smin, smax = spzeros(length(acxmin)), ones(length(acxmax))
  for i in 1:length(acxmin)
    if acxmin[i] * acxmax[i] >= 0
      smin[i] = tanh(acxmax[i]) / acxmax[i]
      smax[i] = tanh(acxmin[i]) / acxmin[i]
    else
      smin[i] = min(tanh(acxmin[i]) / acxmin[i], tanh(acxmax[i]) / acxmax[i])
      smax[i] = 1.0
    end
  end
  return smin, smax
end


