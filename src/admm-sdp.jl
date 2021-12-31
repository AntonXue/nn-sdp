module AdmmSdp

using ..Header
using ..Common
using ..Intervals
using ..Partitions
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
  ξvardims :: Tuple{Int, Int, Vector{Int}}
end

#
@with_kw struct AdmmCache
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
  Dinv :: Vector{Float64}
end

#
function initParams(inst :: SafetyInstance, opts :: AdmmSdpOptions)
  ffnet = inst.ffnet
  input = inst.input
  safety = inst.safety

  γdims, ξvardims = makeγdims(opts.β, inst)
  ξindim, ξsafedim, ξkdims = ξvardims
  @assert ξsafedim == 0
  @assert length(ξkdims) >= 2

  # Initialize the iteration variables
  γ = zeros(sum(γdims))

  num_cliques = ffnet.K - opts.β - 1

  # Each ωk = Hk * γ
  ωs = [zeros(sum(γdims[i] for i in Hinds(k, opts.β, γdims))) for k in 1:num_cliques]

  # Each vk is the size of sum(Ckdim)^2
  vs = [zeros(length(sum(Cdims(k, opts.β, ffnet.zdims))^2)) for k in 1:num_cliques]

  # These are derived from vs and ωs respectively
  λs = [zeros(length(vs[k])) for k in 1:num_cliques]
  μs = [zeros(length(ωs[k])) for k in 1:num_cliques]

  params = AdmmParams(γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs, γdims=γdims, ξvardims=ξvardims)
  return params
end

#
function precomputeCache(params :: AdmmParams, inst :: SafetyInstance, opts :: AdmmSdpOptions)
  precompute_start_time = time()
  if opts.verbose; println("precompute") end

  γdims = params.γdims
  ξvardims = params.ξvardims
  ffnet = inst.ffnet
  num_cliques = length(γdims)

  # Yss[k] is the non-affine components of Yk, Yaffs[k] is the affine component of Yk
  Yss = Vector{Vector{Matrix{Float64}}}()
  Yaffs = Vector{Matrix{Float64}}()
  for k = 1:num_cliques
    Yk_start_time = time()
    γkdim = γdims[k]

    # Need to construct the affine component first
    Ykaff = makeYk(k, opts.β, zeros(γkdim), ξvardims, inst, x_intvs=opts.x_intervals, slope_intvs=opts.slope_intervals)

    # Now do the other parts
    Ykparts = Vector{Matrix{Float64}}()
    for i in 1:γkdim
      tmp = makeYk(k, opts.β, e(i, γkdim), ξvardims, inst, x_intvs=opts.x_intervals, slope_intvs=opts.slope_intervals)
      Yki = tmp - Ykaff
      push!(Ykparts, Yki)
    end

    # Store both the Ykparts and the affine component
    push!(Yss, Ykparts)
    push!(Yaffs, Ykaff)

    Yk_time = round(time() - Yk_start_time, digits=2)
    if opts.verbose; println("\tYss[" * string(k) * "/" * string(num_cliques) * "], time: " * string(Yk_time)) end
  end

  # Compute the other components that are dependent on the Ys
  Js = Vector{Matrix{Float64}}()
  zaffs = Vector{Vector{Float64}}()
  Jtzaffs = Vector{Vector{Float64}}()
  I_JtJ_invs = Vector{Matrix{Float64}}()

  Ωinvs = makeΩinvs(opts.β, ffnet.zdims)

  for k in 1:num_cliques
    Jk_start_time = time()

    # We start off Jk as a list of vectors that we will then hcat together
    Jk = Vector{Vector{Float64}}()

    # We will eventually sum the elements of this to get zkaff
    zkaffparts = Vector{Vector{Float64}}()

    Eck = Ec(k, opts.β, ffnet.zdims)
    for j = -opts.β:opts.β
      if (k+j < 1) || (k+j > num_cliques); continue end

      Eckj = Ec(k+j, opts.β, ffnet.zdims)
      γkjdim = γdims[k+j]
      for Ykji in Yss[k+j]
        insYkji = Eck * Eckj' * (Ωinvs[k+j] .* Ykji) * Eckj * Eck'
        push!(Jk, vec(insYkji))
      end

      insYkjaff = Eck * Eckj' * (Ωinvs[k+j] .* Yaffs[k+j]) * Eckj * Eck'
      push!(zkaffparts, vec(insYkjaff))
    end

    # Finish and store Jk
    Jk = hcat(Jk...)
    push!(Js, Jk)

    # Finish and store zkaff
    zkaff = sum(zkaffparts)
    push!(zaffs, zkaff)

    # Jk' * zkaff
    _Jktzaff = Jk' * zkaff
    push!(Jtzaffs, _Jktzaff)

    # inv(I + Jk' * Jk)
    _I_JktJk_inv = inv(Symmetric(I + Jk' * Jk))
    push!(I_JtJ_invs, _I_JktJk_inv)

    Jk_time = round(time() - Jk_start_time, digits=2)
    if opts.verbose; println("\tJ[" * string(k) * "/" * string(num_cliques) * "], time: " * string(Jk_time)) end
  end

  # Compute Dinv
  D = sum(H(k, opts.β+1, γdims)' * H(k, opts.β+1, γdims) for k in 1:num_cliques)
  D = diag(D)
  @assert minimum(D) >= 1
  Dinv = 1 ./ D

  # Complete
  cache = AdmmCache(Js=Js, zaffs=zaffs, Jtzaffs=Jtzaffs, I_JtJ_invs=I_JtJ_invs, Dinv=Dinv)
  return cache
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


export initParams, precomputeCache
export AdmmSdpOptions, AdmmParams, AdmmCache

end # End module

