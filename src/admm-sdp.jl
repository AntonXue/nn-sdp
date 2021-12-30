module AdmmSdp

using ..Header
using ..Common
using ..Partitions
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using Mosek
using MosekTools

#
@with_kw struct AdmmSdpOptions
  max_iters :: Int = 400
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  β :: Int = 1
  ρ :: Float64 = 1.0
  x_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  verbose :: Bool = false
end

#
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ωs :: Vector{Vector{Float64}}
  λs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}

  # Some auxiliary stuff that we keep around too
  γdims :: Vector{Int}
  ξvardims :: Tuple{Int, Vector{Int}}
end

#
@with_kw struct AdmmCache
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
end


#
function initParams(inst :: SafetyInstance, opts :: AdmmSdpOptions)
  ffnet = inst.ffnet
  input = inst.input
  safety = inst.safety

  num_cliques = ffnet.K - opts.β - 1

  # First calculate the qxdims, which are used to calculate ξdims
  qxdims = [sum(ffnet.zdims[(k+1):(k+opts.β)]) for k in 1:(num_cliques+1)]

  # Use qxdims to calculate the ξvardims
  ξindim = (input isa BoxInput) ? ffnet.xdims[1] : (input isa PolytopeInput ? ffnet.xdims[1]^2 : error(""))
  ξkdims = [qxdims[k]^2 + (4 * qxdims[k]) for k in 1:(num_cliques+1)]
  ξvardims = (ξindim, ξkdims)

  # Use the ξvardims information to calculate γdims
  γdims = Vector{Int}()
  for k = 1:num_cliques
    if num_cliques == 1 && k == 1
      push!(γdims, ξindim + ξkdims[1] + ξkdims[2])
    elseif k == 1
      push!(γdims, ξindim + ξkdims[1])
    elseif k == num_cliques
      push!(γdims, ξkdims[k] + ξkdims[k+1])
    else
      push!(γdims, ξkdims[k])
    end
  end

  # Initialize the iteration variables
  γ = zeros(sum(γdims))

  # Each ωk = Hk * γ
  ωs = [zeros(sum(γdims[i] for i in Hcinds(k, opts.β, γdims))) for k in 1:num_cliques]

  # Each vk is the size of sum(Ckdim)^2
  vs = [zeros(length(sum(Cdims(k, opts.β, ffnet.zdims))^2)) for k in 1:num_cliques]

  # These are derived from vs and ωs respectively
  λs = [zeros(length(vs[k])) for k in 1:num_cliques]
  μs = [zeros(length(ωs[k])) for k in 1:num_cliques]

  return AdmmParams(
    γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs,
    γdims=γdims,
    ξvardims=ξvardims)
end

#
function precompute
end

#
function makezk(k :: Int, ωk :: Vector{Float64}, cache :: AdmmCache)
  return cache.Js[k] * ωk + cache.zaffs[k]
end

#
function projectΓ(γ :: Vector{Float64})
  return max.(γ, 0)
end

# Project a vector onto the negative semidefinite cone
function projectNsd(vk :: Vector{Float64})
  dim = Int(round(sqrt(length(vk)))) # :)
  @assert length(vk) == dim * dim
  tmp = Symmetric(reshape(vk, (dim, dim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

#
function stepγ(params :: AdmmParams, cache :: AdmmCache)
end

#
function stepvk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
end

#
function stepωk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
end

#
function stepλk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
end

#
function stepμk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
end


export AdmmSdpOptions, AdmmParams, AdmmCache

end # End module

