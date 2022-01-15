module AdmmSdp

using ..Header
using ..Common
using ..Intervals
# using ..Partitions
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
  x_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  tband_func :: Function = (k, qkxdim) -> qkxdim
  verbose :: Bool = false
end

# The parameters used by ADMM, as well as some helpful information
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  zs :: Vector{Vector{Float64}}
  λs :: Vector{Vector{Float64}}

  # Some auxiliary stuff that we keep around too
  ξindim :: Int
  ξsafedim :: Int
  ξkdims :: Vector{Int}
  num_cliques :: Int = length(ξkdims) - 1
  @assert num_cliques >= 1
end

# The things that we cache prior to the ADMM steps
@with_kw struct AdmmCache
  J :: Matrix{Float64}
  zaff :: Vector{Float64}; @assert size(J)[1] == length(zaff)
end

# Admm step status
@with_kw struct AdmmSummary
  test :: Int
end

# Initialize zero-valued parameters of the appropriate size
function initParams(inst :: QueryInstance, opts :: AdmmSdpOptions)
  ffnet = inst.ffnet
  input = inst.input

  ξvardims = makeξvardims(opts.β, inst, opts.tband_func)
  ξindim, ξsafedim, ξkdims = ξvardims
  @assert length(ξkdims) >= 2

  # Initialize the iteration variables
  γ = zeros(ξindim + ξsafedim + sum(ξkdims))

  num_cliques = ffnet.K - opts.β - 1
  vdims = [size(Ec(k, opts.β, ffnet.zdims))[1] for k in 1:num_cliques]
  vs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  zs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  λs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  params = AdmmParams(γ=γ, vs=vs, zs=zs, λs=λs, ξindim=ξindim, ξsafedim=ξsafedim, ξkdims=ξkdims)
  return params
end

# Cache precomputation
function precompute(inst :: QueryInstance, params :: AdmmParams, opts :: AdmmSdpOptions)
  @assert inst.ffnet.type isa ReluNetwork

  input = inst.input
  ffnet = inst.ffnet
  zdims = ffnet.zdims

  num_cliques = ffnet.K - opts.β - 1
  @assert num_cliques >= 1

  # Some helpful block matrices
  E1 = E(1, zdims)
  EK = E(ffnet.K, zdims)
  Ea = E(ffnet.K+1, zdims)
  Ein = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Relevant xqinfos
  xqinfos = Vector{Xqinfo}()
  for k = 1:(num_cliques+1)
    qxdim = Qxdim(k, opts.β, zdims)
    xqinfo = Xqinfo(
      ffnet = ffnet,
      ϕout_intv = selectϕoutIntervals(k, opts.β, opts.x_intvs),
      slope_intv = selectSlopeIntervals(k, opts.β, opts.slope_intvs),
      tband = opts.tband_func(k, qxdim))
    push!(xqinfos, xqinfo)
  end

  # Compute the J, but first we need to compute the affine components
  xinaff = vec(Ein' * makeXin(zeros(params.ξindim), input, ffnet) * Ein)

  if inst isa SafetyInstance
    xsafeaff = vec(Esafe' * makeXsafe(inst.safety.S, ffnet) * Esafe)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    S0 = makeShyperplane(inst.reach_set.normal, 0, ffnet)
    xsafeaff = vec(Esafe' * makeXsafe(S0, ffnet) * Esafe)
  else
    error("unsupported instance: " * string(inst))
  end

  xkaffs = Vector{Vector{Float64}}()
  for k = 1:(num_cliques+1)
    EXk = [E(k, opts.β, zdims); Ea]
    qxdim = Qxdim(k, opts.β, zdims)
    xkaff = vec(EXk' * makeXqξ(k, opts.β, zeros(params.ξkdims[k]), xqinfos[k]) * EXk)
    push!(xkaffs, xkaff)
  end

  zaff = xinaff + xsafeaff + sum(xkaffs)

  # Now that we have computed the affine components, can begin actually computing J
  Jparts = Vector{Any}()

  # ... first computing Xin
  for i in 1:params.ξindim
    xini = vec(Ein' * makeXin(e(i, params.ξindim), input, ffnet) * Ein) - xinaff
    push!(Jparts, xini)
  end

  # ... then doing the Xsafe, but currently only applies if we're a reach instance
  if inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    S1 = makeShyperplane(inst.reach_set.normal, 1, ffnet)
    xsafe1 = vec(Esafe' * makeXsafe(S1, ffnet) * Esafe) - xsafeaff
    push!(Jparts, xsafe1)
  end

  # ... and finally the Xks
  for k in 1:(num_cliques+1)
    EXk = [E(k, opts.β, zdims); Ea]
    for i in 1:params.ξkdims[k]
      xki = vec(EXk' * makeXqξ(k, opts.β, e(i, params.ξkdims[k]), xqinfos[k]) * EXk) - xkaffs[k]
      push!(Jparts, xki)
    end
  end

  J = hcat(Jparts...)

  cache = AdmmCache(J=J, zaff=zaff)
  return cache
end

# Caching process to be run before the ADMM iterations
function makezβ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  return cache.J * params.γ + cache.zaff
end

# Make a nonnegative projection
function projectNonnegative(γ :: Vector{Float64})
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
      if opts.verbose; println("failed check pair (k, nsdtol): " * string((k, eigmax(tmp)))) end
      return false
    end
  end
  if opts.verbose; println("SAT!") end
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
  start_params = initParams(inst, opts)
  cache, setup_time = precomputeCache(start_params, inst, opts)

  final_params, admm_time = admm(start_params, cache, opts)
  total_time = time() - start_time
    
  
  status = isγSat(final_params, cache, opts) ? "OPTIMAL" : "INFEASIBLE"


  return SolutionOutput(
    objective_value = 0.0,
    values = final_params,
    summary = (),
    termination_status = status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = admm_time)
end

export initParams, precomputeCache
export AdmmSdpOptions, AdmmParams, AdmmCache

end # End module

