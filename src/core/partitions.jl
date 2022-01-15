# ξ-based partitioning
module Partitions

using ..Header
using ..Common
using ..Intervals
using Parameters

# Splice a vector
function splice(x, sizes :: Vector{Int})
  @assert all(sizes .>= 0)
  @assert 1 <= length(x) == sum(sizes)
  num_sizes = length(sizes)
  highs = [sum(sizes[1:k]) for k in 1:num_sizes]
  lows = [1; [1 + highk for highk in highs[1:end-1]]]
  @assert length(highs) == length(lows)
  splices = [x[lows[k] : highs[k]] for k in 1:num_sizes]
  return splices
end

function makeξdims(b :: Int, inst :: QueryInstance, tband_func :: Function)
  @assert inst isa SafetyInstance || inst isa ReachabilityInstance
  @assert inst.ffnet.type isa ReluNetwork

  # Set up the ξindim
  if inst.input isa BoxInput
    ξindim = inst.ffnet.xdims[1]
  elseif inst.input isa PolytopeInput
    ξindim = inst.ffnet.xdims[1]^2
  else
    error("unsupported input: " * string(inst.input))
  end

  # Set up the ξsafedim
  if inst isa SafetyInstance
    ξsafedim = 0
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    ξsafedim = 1
  else
    error("unsupported instance: " * string(inst))
  end

  # Set up the ξkdims
  zdims = inst.ffnet.zdims
  num_cliques = length(zdims) - b - 2
  ξkdims = Vector{Int}()
  if inst.ffnet.type isa ReluNetwork
    for k = 1:(num_cliques+1)
      qxdim = Qxdim(k, b, zdims)
      tband = tband_func(k, qxdim)
      λ_slope_length = λlength(qxdim, tband)
      η_slope_length = qxdim
      ν_slope_length = qxdim
      d_out_length = qxdim
      ξkdim = λ_slope_length + η_slope_length + ν_slope_length + d_out_length
      push!(ξkdims, ξkdim)
    end
  else
    error("unsupported network: " * string(inst.ffnet))
  end

  return ξindim, ξsafedim, ξkdims
end

# Calculate the dimensions of the ξk variables
function makeξvardims(b :: Int, zdims :: Vector{Int}, tband_func :: Function; ξindim :: Int = 0, ξsafedim :: Int = 0)
  @assert ξindim >= 1 && ξsafedim >= 0
  num_cliques = length(zdims) - b - 2

  ξkdims = Vector{Int}()
  for k = 1:(num_cliques+1)
    qxdim = Qxdim(k, b, zdims)
    tband = tband_func(k, qxdim)
    λ_slope_length = λlength(qxdim, tband)
    η_slope_length = qxdim
    ν_slope_length = qxdim
    d_out_length = qxdim
    ξkdim = λ_slope_length + η_slope_length + ν_slope_length + d_out_length
    push!(ξkdims, ξkdim)
  end

  ξvardims = (ξindim, ξsafedim, ξkdims)
  return ξvardims
end

# Figure out the appropriate γdims
function makeγdims(b :: Int, inst :: QueryInstance, tband_func :: Function)
  @assert inst isa SafetyInstance || inst isa ReachabilityInstance
  if inst.input isa BoxInput
    ξindim = inst.ffnet.xdims[1]
  elseif inst.input isa PolytopeInputo
    ξindim = inst.ffnet.xdims[1]^2
  else
    error("unsupported input: " * string(inst.input))
  end

  if inst isa SafetyInstance
    ξsafedim = 0
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    ξsafedim = 1
  else
    error("unsupported instance: " * string(inst))
  end

  ξvardims = makeξvardims(b, inst.ffnet.zdims, tband_func, ξindim=ξindim, ξsafedim=ξsafedim)
  _, _, ξkdims = ξvardims

  if length(ξkdims) == 2
    γ1dim = ξindim + ξsafedim + sum(ξkdims)
    γdims = [γ1dim]
    return γdims, ξvardims

  else
    γ1dim = ξindim + ξsafedim + ξkdims[1]
    γpdim = ξkdims[end-1] + ξkdims[end]
    γkdims = [ξkdim for ξkdim in ξkdims[2:end-2]]
    γdims = [γ1dim; γkdims; γpdim]
    return γdims, ξvardims
  end
end

# Split the γ1 component
function spliceγ1(γ1, ξvardims)
  ξindim, ξsafedim, ξkdims = ξvardims
  @assert length(ξkdims) >= 2

  # When there is only 1 clique
  if length(ξkdims) == 2
    return splice(γ1, [ξindim; ξsafedim; ξkdims[1]; ξkdims[2]])

  # When there are > 1 clique
  else
    return splice(γ1, [ξindim; ξsafedim; ξkdims[1]])
  end
end

# The indices k-b ... k+b involved in an H projection
function Hinds(k :: Int, b :: Int, γdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k <= length(γdims)
  return [k+j for j in -b:b if 1 <= k+j <= length(γdims)]
end

# The Hk projection matrix
function H(k :: Int, b :: Int, γdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k <= length(γdims)
  inds = Hinds(k, b, γdims)
  Eks = [E(i, γdims) for i in inds]
  return vcat(Eks...)
end

# Indexed version to calculate ωk = Hk * γ
function indexedH(k :: Int, b :: Int, γdims :: Vector{Int}, γ)
  @assert k >= 1 && b >= 1
  @assert 1 <= k <= length(γdims)
  @assert length(γ) == sum(γdims)
  low, high = max(k-b, 1), min(k+b, length(γdims))
  start = sum(γdims[1:(low-1)]) + 1
  final = sum(γdims[1:high])
  return γ[start:final]
end

# Indexed version to calculate Hk' * x
function indexedHt(k :: Int, b :: Int, γdims :: Vector{Int}, x)
  @assert k >= 1 && b >= 1
  @assert 1 <= k <= length(γdims)
  low, high = max(k-b, 1), min(k+b, length(γdims))
  start = sum(γdims[1:(low-1)]) + 1
  final = sum(γdims[1:high])
  @assert length(x) == final - start + 1
  Hktx = zeros(sum(γdims))
  Hktx[start:final] = x
  return Hktx
end

# The version of makeXq that takes a vector ξk
function makeXqξ(k :: Int, b :: Int, ξk, xqinfo :: Xqinfo)
  ffnet = xqinfo.ffnet
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K

  qxdim = Qxdim(k, b, ffnet.zdims)
  if ffnet.type isa ReluNetwork
    # Decompose the ξk
    λ_slope_start = 1
    λ_slope_final = λlength(qxdim, xqinfo.tband)
    η_slope_start = λ_slope_final + 1
    η_slope_final = λ_slope_final + qxdim
    ν_slope_start = η_slope_final + 1
    ν_slope_final = η_slope_final + qxdim
    d_out_start = ν_slope_final + 1
    d_out_final = ν_slope_final + qxdim

    @assert length(ξk) == d_out_final
    λ_slope = ξk[λ_slope_start:λ_slope_final]
    η_slope = ξk[η_slope_start:η_slope_final]
    ν_slope = ξk[ν_slope_start:ν_slope_final]
    d_out = ξk[d_out_start:d_out_final]
    vars = (λ_slope, η_slope, ν_slope, d_out)
    return makeXq(k, b, vars, xqinfo)
  else
    error("unsupported network: " * string(ffnet))
  end
end

# The version of makeXin that takes a vector ξin
function makeXinξ(ξin, input :: InputConstraint, ffnet :: FeedForwardNetwork)
  return makeXin(ξin, input, ffnet)
end

# Make a safety version of Xsafe
function makeSafetyXsafeξ(safety :: SafetyConstraint, ffnet :: FeedForwardNetwork)
  return makeXsafe(safety.S, ffnet)
end

# Make the hyperplane reachability version of Xsafe
function makeHyperplaneReachXsafeξ(ξsafe, hplane :: HyperplaneSet, ffnet :: FeedForwardNetwork)
  S = makeShyperplane(hplane.normal, ξsafe, ffnet)
  return makeXsafe(S, ffnet)
end

# Some information that helps constrct the Ys
@with_kw struct Yinfo
  inst :: QueryInstance
  num_cliques :: Int
  x_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  ξvardims :: Tuple{Int, Int, Vector{Int}}
  tband_func :: Function
end

# Make Y1 when there is only 1 clique
function _makeOnlyY1(b :: Int, γ1, Ckdims, yinfo :: Yinfo)
  ξindim, ξsafedim, ξkdims = yinfo.ξvardims
  spliced = splice(γ1, [ξindim; ξsafedim; ξkdims[1]; ξkdims[2]])
  ξin, ξsafe, ξ1, ξ2 = spliced[1], spliced[2], spliced[3], spliced[4]

  inst = yinfo.inst
  if inst isa SafetyInstance
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeSafetyXsafeξ(inst.safety, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeHyperplaneReachXsafeξ(ξsafe, inst.reach_set, inst.ffnet)
  else
    error("unrecognized instance: " * string(inst))
  end

  q1xdim = Qxdim(1, b, yinfo.inst.ffnet.zdims)
  xq1info = Xqinfo(
    ffnet = inst.ffnet,
    ϕout_intv = selectϕoutIntervals(1, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(1, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(1, q1xdim))
  X1 = makeXqξ(1, b, ξ1, xq1info)

  q2xdim = Qxdim(2, b, yinfo.inst.ffnet.zdims)
  xq2info = Xqinfo(
    ffnet = inst.ffnet,
    ϕout_intv = selectϕoutIntervals(2, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(2, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(2, q2xdim))
  X2 = makeXqξ(2, b, ξ2, xq2info)

  # Set up the intra-clique selectors
  F1 = E(1, Ckdims)
  FK = E(length(Ckdims)-1, Ckdims)
  Faff = E(length(Ckdims), Ckdims)
  Fin = [F1; Faff]
  Fsafe = [F1; FK; Faff]
  FX1 = [E(1, b, Ckdims); Faff]
  FX2 = [E(2, b, Ckdims); Faff]
  Y1 = (Fin' * Xin * Fin) + (Fsafe' * Xsafe * Fsafe) + (FX1' * X1 * FX1) + (FX2' * X2 * FX2)
  return Y1
end

# Make Y1 when there is > 1 clique
function _makeY1(b :: Int, γ1, Ckdims, yinfo :: Yinfo)
  ξindim, ξsafedim, ξkdims = yinfo.ξvardims
  spliced = splice(γ1, [ξindim; ξsafedim; ξkdims[1]])
  ξin, ξsafe, ξ1 = spliced[1], spliced[2], spliced[3]

  inst = yinfo.inst
  if inst isa SafetyInstance
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeSafetyXsafeξ(inst.safety, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeHyperplaneReachXsafeξ(ξsafe, inst.reach_set, inst.ffnet)
  else
    error("unrecognized instance: " * string(inst))
  end

  q1dim = Qxdim(1, b, yinfo.inst.ffnet.zdims)
  xq1info = Xqinfo(
    ffnet = inst.ffnet,
    ϕout_intv = selectϕoutIntervals(1, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(1, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(1, q1dim))
  X1 = makeXqξ(1, b, ξ1, xq1info)

  # Set up the intra-clique selectors
  F1 = E(1, Ckdims)
  FK = E(length(Ckdims)-1, Ckdims)
  Faff = E(length(Ckdims), Ckdims)
  Fin = [F1; Faff]
  Fsafe = [F1; FK; Faff]
  FX1 = [E(1, b, Ckdims); Faff]
  Y1 = (Fin' * Xin * Fin) + (Fsafe' * Xsafe * Fsafe) + (FX1' * X1 * FX1)
  return Y1
end

# Make the final Yp
function _makeYp(p :: Int, b :: Int, γp, Ckdims, yinfo :: Yinfo)
  _, _, ξkdims = yinfo.ξvardims
  spliced = splice(γp, [ξkdims[end-1], ξkdims[end]])
  ξp, ξq = spliced[1], spliced[2]

  qpxdim = Qxdim(p, b, yinfo.inst.ffnet.zdims)
  xqpinfo = Xqinfo(
    ffnet = yinfo.inst.ffnet,
    ϕout_intv = selectϕoutIntervals(p, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(p, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(p, qpxdim))
  Xp = makeXqξ(p, b, ξp, xqpinfo)

  q = p + 1
  qqxdim = Qxdim(q, b, yinfo.inst.ffnet.zdims)
  xqqinfo = Xqinfo(
    ffnet = yinfo.inst.ffnet,
    ϕout_intv = selectϕoutIntervals(q, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(q, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(q, qqxdim))
  Xq = makeXqξ(q, b, ξq, xqqinfo)

  # Set up the intra-clique selectors
  Faff = E(length(Ckdims), Ckdims)
  FXp = [E(1, b, Ckdims); Faff]
  FXq = [E(2, b, Ckdims); Faff]
  Yp = (FXp' * Xp * FXp) + (FXq' * Xq * FXq)
  return Yp
end

# Make an Yk when k is neither 1 nor p
function _makeYk(k :: Int, b :: Int, γk, Ckdims, yinfo :: Yinfo)
  ξk = γk
  qkxdim = Qxdim(k, b, yinfo.inst.ffnet.zdims)
  xqkinfo = Xqinfo(
    ffnet = yinfo.inst.ffnet,
    ϕout_intv = selectϕoutIntervals(k, b, yinfo.x_intvs),
    slope_intv = selectSlopeIntervals(k, b, yinfo.slope_intvs),
    tband = yinfo.tband_func(k, qkxdim))
  Xk = makeXqξ(k, b, ξk, xqkinfo)

  # Set up the intra-clique selectors
  Faff = E(length(Ckdims), Ckdims)
  FXk = [E(1, b, Ckdims); Faff]
  Yk = FXk' * Xk * FXk
  return Yk
end

# Call this

function makeYk(k :: Int, b :: Int, γk, yinfo :: Yinfo)
  # The dimension components of this clique
  Ckdims = Cdims(k, b, yinfo.inst.ffnet.zdims)

  # Since this is the only clique, contains everything
  if yinfo.num_cliques == 1
    return _makeOnlyY1(b, γk, Ckdims, yinfo)

  # More than one clique; Y1
  elseif k == 1
    return _makeY1(b, γk, Ckdims, yinfo)

  # More than one clique; Yp
  elseif k == yinfo.num_cliques
    return _makeYp(k, b, γk, Ckdims, yinfo)

  # Otherwise some intermediate Yk
  else
    return _makeYk(k, b, γk, Ckdims, yinfo)
  end
end

# Each size(Ωk) == size(Yk)
# The entry (Ωk)_{ij} counts the times (Yk)_{ij} is used in different Z1 ... Zp's
function makeΩs(b :: Int, zdims :: Vector{Int})
  @assert 1 <= length(zdims) - b - 2
  @assert zdims[end] == 1
  num_cliques = length(zdims) - b - 2
  Zdim = sum(zdims)
  Ωs = Vector{Any}()
  for k in 1:num_cliques
    Sk = zeros(Zdim, Zdim)
    # Since Zk depends on at most Y[k-b] ... Y[k+b], this narrows our search
    for j in -b:b
      if (k+j < 1) || (k+j > num_cliques); continue end
      Eckj = Ec(k+j, b, zdims)
      ckjdim = size(Eckj)[1]
      Sk = Sk + (Eckj' * ones(ckjdim, ckjdim) * Eckj)
    end
    Eck = Ec(k, b, zdims)
    Ωk = Eck * Sk * Eck'
    push!(Ωs, Ωk)
  end
  return Ωs
end

# Semantically Ωinv = ℧, but the rendering of ℧ just looks so ugly here
function makeΩinvs(b :: Int, zdims :: Vector{Int})
  @assert 1 <= length(zdims) - b - 2
  @assert zdims[end] == 1
  num_cliques = length(zdims) - b - 2
  Ωs = makeΩs(b, zdims)
  @assert length(Ωs) == num_cliques
  Ωinvs = Vector{Matrix{Float64}}()
  for k in 1:num_cliques
    Ωkinv = 1 ./ Ωs[k]
    Ωkinv[isinf.(Ωkinv)] .= 0
    push!(Ωinvs, Ωkinv)
  end
  return Ωinvs
end

# Make Zk from a bunch of Ys and Ωinvs
function makeZk(k :: Int, b :: Int, Ys, Ωinvs :: Vector{Matrix{Float64}}, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  num_cliques = length(zdims) - b - 2
  @assert 1 <= num_cliques == length(Ωinvs)
  Zdim = sum(zdims)
  Ysum = zeros(Zdim, Zdim)
  for j = -b:b
    if (k+j < 1) || (k+j > num_cliques); continue end
    Eckj = Ec(k+j, b, zdims)
    Ysum = Ysum + (Eckj' * (Ωinvs[k+j] .* Ys[k+j]) * Eckj)
  end
  Eck = Ec(k, b, zdims)
  Zk = Eck * Ysum * Eck'
  return Zk
end

#
export splice, spliceγ1
export makeξdims
export makeξvardims, makeγdims
export Hinds, H, indexedH, indexedHt
export makeXqξ, makeXinξ, makeSafetyXsafeξ, makeHyperplaneReachXsafeξ
export Yinfo
export makeYk
export makeΩs, makeΩinvs
export makeZk

end # End module

