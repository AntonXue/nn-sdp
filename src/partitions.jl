# ξ-based partitioning
module Partitions

using ..Header
using ..Common
using ..Intervals

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

# Calculate the dimensions of the ξvars
function makeξvardims(b :: Int, zdims :: Vector{Int}; ξindim :: Int = 0, ξsafedim :: Int = 0)
  @assert ξindim >= 1 && ξsafedim >= 0
  num_cliques = length(zdims) - b - 2
  @assert num_cliques >= 1
  qxdims = [sum(zdims[(k+1):(k+b)]) for k in 1:(num_cliques+1)]
  ξkdims = [qxdims[k]^2 + (4 * qxdims[k]) for k in 1:(num_cliques+1)]
  ξvardims = (ξindim, ξsafedim, ξkdims)
  return ξvardims
end

# Figure out the appropriate γdims
function makeγdims(b :: Int, inst :: QueryInstance)
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

  ξvardims = makeξvardims(b, inst.ffnet.zdims, ξindim=ξindim, ξsafedim=ξsafedim)
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

  # There is only 1 clique
  if length(ξkdims) == 2
    return splice(γ1, [ξindim; ξsafedim; ξkdims[1]; ξkdims[2]])

  # There is > 1 clique
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

# The version of makeXq that takes a vector ξk
function makeXqξ(k :: Int, b :: Int, ξk, ffnet :: FeedForwardNetwork; ϕout_intv = nothing, slope_intv = nothing)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K

  qxdim = sum(ffnet.zdims[k+1:k+b])
  if ffnet.type isa ReluNetwork
    # Decompose the ξ
    λ_slope_start = 1
    λ_slope_final = qxdim
    τ_slope_start = λ_slope_final + 1
    τ_slope_final = λ_slope_final + (qxdim)^2
    η_slope_start = τ_slope_final + 1
    η_slope_final = τ_slope_final + qxdim
    ν_slope_start = η_slope_final + 1
    ν_slope_final = η_slope_final + qxdim
    d_out_start = ν_slope_final + 1
    d_out_final = ν_slope_final + qxdim

    @assert length(ξk) == d_out_final
    λ_slope = ξk[λ_slope_start:λ_slope_final]
    τ_slope = reshape(ξk[τ_slope_start:τ_slope_final], (qxdim, qxdim))
    η_slope = ξk[η_slope_start:η_slope_final]
    ν_slope = ξk[ν_slope_start:ν_slope_final]
    d_out = ξk[d_out_start:d_out_final]
    vars = (λ_slope, τ_slope, η_slope, ν_slope, d_out)
    return makeXq(k, b, vars, ffnet, ϕout_intv=ϕout_intv, slope_intv=slope_intv)
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

# Make Y1 when there is only 1 clique
function _makeOnlyY1(b :: Int, γ1, ξvardims, Ckdims, inst; x_intvs = nothing, slope_intvs = nothing) 
  ξindim, ξsafedim, ξkdims = ξvardims
  spliced = splice(γ1, [ξindim; ξsafedim; ξkdims[1]; ξkdims[2]])
  ξin, ξsafe, ξ1, ξ2 = spliced[1], spliced[2], spliced[3], spliced[4]

  if inst isa SafetyInstance
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeSafetyXsafeξ(inst.safety, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeHyperplaneReachXsafeξ(ξsafe, inst.reach_set, inst.ffnet)
  else
    error("unrecognized instance: " * string(inst))
  end

  ϕout_intv1 = selectϕoutIntervals(1, b, x_intvs)
  slope_intv1 = selectSlopeIntervals(1, b, slope_intvs)
  X1 = makeXqξ(1, b, ξ1, inst.ffnet, ϕout_intv=ϕout_intv1, slope_intv=slope_intv1)

  ϕout_intv2 = selectϕoutIntervals(2, b, x_intvs)
  slope_intv2 = selectSlopeIntervals(2, b, slope_intvs)
  X2 = makeXqξ(2, b, ξ2, inst.ffnet, ϕout_intv=ϕout_intv2, slope_intv=slope_intv2)

  # Set up the intra-clique selectors
  F1 = E(1, Ckdims)
  FK = E(length(Ckdims)-1, Ckdims)
  Faff = E(length(Ckdims), Ckdims)
  Fin = [F1; Faff]
  Fsafe = [F1; FK; Faff]
  FX1 = [E(1, b, Ckdims); Faff]
  FX2 = [E(2, b, Ckdims); Faff]
  Yk = (Fin' * Xin * Fin) + (Fsafe' * Xsafe * Fsafe) + (FX1' * X1 * FX1) + (FX2' * X2 * FX2)
  return Yk
end

# Make Y1 when there is > 1 clique
function _makeY1(b :: Int, γ1, ξvardims, Ckdims, inst; x_intvs = nothing, slope_intvs = nothing)
  ξindim, ξsafedim, ξkdims = ξvardims
  spliced = splice(γ1, [ξindim; ξsafedim; ξkdims[1]])
  ξin, ξsafe, ξ1 = spliced[1], spliced[2], spliced[3]

  if inst isa SafetyInstance
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeSafetyXsafeξ(inst.safety, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    Xin = makeXinξ(ξin, inst.input, inst.ffnet)
    Xsafe = makeHyperplaneReachXsafeξ(ξsafe, inst.reach_set, inst.ffnet)
  else
    error("unrecognized instance: " * string(inst))
  end

  ϕout_intv = selectϕoutIntervals(1, b, x_intvs)
  slope_intv = selectSlopeIntervals(1, b, slope_intvs)
  X1 = makeXqξ(1, b, ξ1, inst.ffnet, ϕout_intv=ϕout_intv, slope_intv=slope_intv)

  # Set up the intra-clique selectors
  F1 = E(1, Ckdims)
  FK = E(length(Ckdims)-1, Ckdims)
  Faff = E(length(Ckdims), Ckdims)
  Fin = [F1; Faff]
  Fsafe = [F1; FK; Faff]
  FX1 = [E(1, b, Ckdims); Faff]
  Yk = (Fin' * Xin * Fin) + (Fsafe' * Xsafe * Fsafe) + (FX1' * X1 * FX1)
  return Yk
end

# Make the final Yp
function _makeYp(k :: Int, b :: Int, γp, ξvardims, Ckdims, inst; x_intvs = nothing, slope_intvs = nothing)
  _, _, ξkdims = ξvardims
  spliced = splice(γp, [ξkdims[end-1], ξkdims[end]])
  ξp, ξq = spliced[1], spliced[2]

  ϕout_intvp = selectϕoutIntervals(k, b, x_intvs)
  slope_intvp = selectSlopeIntervals(k, b, slope_intvs)
  Xp = makeXqξ(k, b, ξp, inst.ffnet, ϕout_intv=ϕout_intvp, slope_intv=slope_intvp)

  ϕout_intvq = selectϕoutIntervals(k+1, b, x_intvs)
  slope_intvq = selectSlopeIntervals(k+1, b, slope_intvs)
  Xq = makeXqξ(k+1, b, ξq, inst.ffnet, ϕout_intv=ϕout_intvq, slope_intv=slope_intvq)

  # Set up the intra-clique selectors
  Faff = E(length(Ckdims), Ckdims)
  FXp = [E(1, b, Ckdims); Faff]
  FXq = [E(2, b, Ckdims); Faff]
  Yk = (FXp' * Xp * FXp) + (FXq' * Xq * FXq)
  return Yk
end

# Make an Yk when k is neither 1 nor p
function _makeYk(k :: Int, b :: Int, γk, ξvardims, Ckdims, inst; x_intvs = nothing, slope_intvs = nothing)
  ξk = γk
  ϕout_intv = selectϕoutIntervals(k, b, x_intvs)
  slope_intv = selectSlopeIntervals(k, b, slope_intvs)
  Xk = makeXqξ(k, b, ξk, inst.ffnet, ϕout_intv=ϕout_intv, slope_intv=slope_intv)

  # Set up the intra-clique selectors
  Faff = E(length(Ckdims), Ckdims)
  FXk = [E(1, b, Ckdims); Faff]
  Yk = FXk' * Xk * FXk
  return Yk
end

# Call this
function makeYk(k :: Int, b :: Int, γk, ξvardims, inst :: QueryInstance; x_intvs = nothing, slope_intvs = nothing)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= inst.ffnet.K - 1
  num_cliques = inst.ffnet.K - b - 1
  @assert num_cliques >= 1

  # The dimension components of this clique
  Ckdims = Cdims(k, b, inst.ffnet.zdims)

  # Since this is the only clique, contains everything
  if num_cliques == 1
    return _makeOnlyY1(b, γk, ξvardims, Ckdims, inst, x_intvs=x_intvs, slope_intvs=slope_intvs)

  # More than one clique; Y1
  elseif k == 1
    return _makeY1(b, γk, ξvardims, Ckdims, inst, x_intvs=x_intvs, slope_intvs=slope_intvs)

  # More than one clique; Yp
  elseif k == num_cliques
    return _makeYp(k, b, γk, ξvardims, Ckdims, inst, x_intvs=x_intvs, slope_intvs=slope_intvs)

  # Otherwise some intermediate Yk
  else
    return _makeYk(k, b, γk, ξvardims, Ckdims, inst, x_intvs=x_intvs, slope_intvs=slope_intvs)
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
  Ωinvs = Vector{Any}()
  for k in 1:num_cliques
    Ωkinv = 1 ./ Ωs[k]
    Ωkinv[isinf.(Ωkinv)] .= 0
    push!(Ωinvs, Ωkinv)
  end
  return Ωinvs
end

# Make Zk from a bunch of Ys and Ωinvs
function makeZk(k :: Int, b :: Int, Ys, Ωinvs, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  num_cliques = length(zdims) - b - 2
  @assert num_cliques >= 1
  Zdim = sum(zdims)
  Ysum = zeros(Zdim, Zdim)
  println("making Z" * string(k) * " with stride " * string(b))
  for j = -b:b
    if (k+j < 1) || (k+j > num_cliques); continue end
    println("\tusing k, j pair: " * string((k, j)))

    Eckj = Ec(k+j, b, zdims)
    Ysum = Ysum + (Eckj' * (Ωinvs[k+j] .* Ys[k+j]) * Eckj)
  end
  Eck = Ec(k, b, zdims)
  Zk = Eck * Ysum * Eck'
  return Zk
end


#
export splice, spliceγ1
export makeξvardims, makeγdims
export Hinds, H
export makeXqξ, makeXinξ, makeSafetyXsafeξ, makeHyperplaneReachXsafeξ
export makeYk
export makeΩs, makeΩinvs
export makeZk

end # End module

