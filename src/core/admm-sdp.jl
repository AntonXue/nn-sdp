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

# The options for ADMM
@with_kw struct AdmmSdpOptions
  max_iters :: Int = 200
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  β :: Int = 1
  ρ :: Float64 = 1.0
  x_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  verbose :: Bool = false
end

# The parameters used by ADMM, as well as some helpful information
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ωs :: Vector{Vector{Float64}}
  λs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}

  # Some auxiliary stuff that we keep around too
  γdims :: Vector{Int}
  ξvardims :: Tuple{Int, Int, Vector{Int}}
  num_cliques :: Int = length(γdims)
end

# The things that we cache prior to the ADMM steps
@with_kw struct AdmmCache
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
  Dinv :: Vector{Float64}
end

# Admm step status
#=
@with_kw struct AdmmSummary
end
=#

# Initialize zero-valued parameters of the appropriate size
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
  vs = [zeros(sum(Cdims(k, opts.β, ffnet.zdims))^2) for k in 1:num_cliques]

  # These are derived from vs and ωs respectively
  λs = [zeros(length(vs[k])) for k in 1:num_cliques]
  μs = [zeros(length(ωs[k])) for k in 1:num_cliques]

  params = AdmmParams(γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs, γdims=γdims, ξvardims=ξvardims)
  return params
end

# Caching process to be run before the ADMM iterations
function precomputeCache(params :: AdmmParams, inst :: SafetyInstance, opts :: AdmmSdpOptions)
  precompute_start_time = time()
  if opts.verbose; println("precompute start") end

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

    Yk_time = round(time() - Yk_start_time, digits=3)
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

    Jk_time = round(time() - Jk_start_time, digits=3)
    if opts.verbose; println("\tJ[" * string(k) * "/" * string(num_cliques) * "], time: " * string(Jk_time)) end
  end

  # Compute Dinv
  D = sum(H(k, opts.β, γdims)' * H(k, opts.β, γdims) for k in 1:num_cliques)
  D = diag(D)
  @assert minimum(D) >= 1
  Dinv = 1 ./ D

  # Complete
  cache = AdmmCache(Js=Js, zaffs=zaffs, Jtzaffs=Jtzaffs, I_JtJ_invs=I_JtJ_invs, Dinv=Dinv)
  precompute_time = round(time() - precompute_start_time, digits=3)
  if opts.verbose; println("precompute time: " * string(precompute_time)) end
  return cache, precompute_time
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
function stepγ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  # tmp = [H(k, opts.β, params.γdims)' * (params.ωs[k] + (params.μs[k] / opts.ρ)) for k in 1:params.num_cliques]
  tmp = [indexedHt(k, opts.β, params.γdims, params.ωs[k] + (params.μs[k] / opts.ρ)) for k in 1:params.num_cliques]
  tmp = sum(tmp)
  tmp = cache.Dinv .* tmp
  return projectΓ(tmp)
end

#
function stepvk(k :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  tmp = makezk(k, params.ωs[k], cache)
  tmp = tmp - (params.λs[k] / opts.ρ)
  return projectNsd(tmp)
end

#
function stepωk(k :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  tmp = cache.Js[k]' * params.vs[k]
  # tmp = tmp + H(k, opts.β, params.γdims) * params.γ
  tmp = tmp + indexedH(k, opts.β, params.γdims, params.γ)
  tmp = tmp + (cache.Js[k]' * params.λs[k] - params.μs[k]) / opts.ρ
  tmp = tmp - cache.Jtzaffs[k]
  tmp = cache.I_JtJ_invs[k] * tmp
  return tmp
end

#
function stepλk(k :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  tmp = params.vs[k] - makezk(k, params.ωs[k], cache)
  tmp = params.λs[k] + opts.ρ * tmp
  return tmp
end

#
function stepμk(k :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  # tmp = params.ωs[k] - H(k, opts.β, params.γdims) * params.γ
  tmp = params.ωs[k] - indexedH(k, opts.β, params.γdims, params.γ)
  tmp = params.μs[k] + opts.ρ * tmp
  return tmp
end

# X = {γ, v1, ..., vp}
function stepX(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_γ = stepγ(params, cache, opts)
  new_vs = Vector([stepvk(k, params, cache, opts) for k in 1:params.num_cliques])
  return new_γ, new_vs
end


# Y = {ω1, ..., ωk}
function stepY(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_ωs = Vector([stepωk(k, params, cache, opts) for k in 1:params.num_cliques])
  return new_ωs
end


# Z = {λ1, ..., λp, μ1, ..., μp}
function stepZ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_λs = Vector([stepλk(k, params, cache, opts) for k in 1:params.num_cliques])
  new_μs = Vector([stepμk(k, params, cache, opts) for k in 1:params.num_cliques])
  return new_λs, new_μs
end

# Residual values
function primalResidual(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  vz_diffs = [params.vs[k] - makezk(k, params.ωs[k], cache) for k in 1:params.num_cliques]
  # ωγ_diffs = [params.ωs[k] - H(k, opts.β, params.γdims) * params.γ for k in 1:params.num_cliques]
  ωγ_diffs = [params.ωs[k] - indexedH(k, opts.β, params.γdims, params.γ) for k in 1:params.num_cliques]
  vz_norm2 = sum(norm(vz_diffs[k])^2 for k in 1:params.num_cliques)
  ωγ_norm2 = sum(norm(ωγ_diffs[k])^2 for k in 1:params.num_cliques)
  norm2 = vz_norm2 + ωγ_norm2
  return vz_diffs, ωγ_diffs, norm2
end

#
function isγSat(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  # for k in 1:params.num_cliques
  for k in reverse(1:params.num_cliques)
    # ωk = H(k, opts.β, params.γdims) * params.γ
    ωk = indexedH(k, opts.β, params.γdims, params.γ)
    zk = makezk(k, ωk, cache)
    dim = Int(round(sqrt(length(zk))))
    tmp = Symmetric(reshape(zk, (dim, dim)))
    if eigmax(tmp) > opts.nsd_tol
      println("failed check pair (k, nsdtol): " * string((k, eigmax(tmp))))
      return false
    end
  end
  println("SAT!")
  return true
end

#
function shouldStop(t :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  if t > opts.begin_check_at_iter && mod(t, opts.check_every_k_iters) == 0
    if isγSat(params, cache, opts); return true end
  end
  return false
end

#
function admm(_params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  iters_run = 0
  total_time = 0
  iter_params = deepcopy(_params)

  for t = 1:opts.max_iters
    step_start_time = time()

    # X stuff
    x_start_time = time()
    new_γ, new_vs = stepX(iter_params, cache, opts)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    x_time = time() - x_start_time

    # Y stuff
    y_start_time = time()
    new_ωs = stepY(iter_params, cache, opts)
    iter_params.ωs = new_ωs
    y_time = time() - y_start_time

    # Z stuff
    z_start_time = time()
    new_λs, new_μs = stepZ(iter_params, cache, opts)
    iter_params.λs = new_λs
    iter_params.μs = new_μs
    z_time = time() - z_start_time

    # Primal residual
    _, _, presidual = primalResidual(iter_params, cache, opts)

    # Coalesce time statistics
    step_time = time() - step_start_time
    total_time = total_time + step_time
    all_times = round.((x_time, y_time, z_time, step_time, total_time), digits=3)

    if opts.verbose
      println("step[" * string(t) * "/" * string(opts.max_iters) * "] times: " * string(all_times))
      println("\tprimal residual: " * string(presidual))
    end

    if shouldStop(t, iter_params, cache, opts); break end
  end

  return iter_params, total_time
end

# Call this
function run(inst :: SafetyInstance, opts :: AdmmSdpOptions)
  start_time = time()
  init_params = initParams(inst, opts)
  cache, setup_time = precomputeCache(init_params, inst, opts)

  final_params, admm_time = admm(start_params, cache, opts)
  total_time = time() - start_time
    

end

export initParams, precomputeCache
export AdmmSdpOptions, AdmmParams, AdmmCache

end # End module

