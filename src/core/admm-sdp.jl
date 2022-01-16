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
  γindim :: Int
  γoutdim :: Int
  γkdims :: Vector{Int}
  num_cliques :: Int = length(γkdims) - 1
  @assert num_cliques >= 1
end

# The things that we cache prior to the ADMM steps
@with_kw struct AdmmCache
  J :: Matrix{Float64}
  zaff :: Vector{Float64}; @assert size(J)[1] == length(zaff)
  Hs :: Vector{Matrix{Float64}}
end

# Admm step status
@with_kw struct AdmmSummary
  test :: Int
end

# Initialize zero-valued parameters of the appropriate size
function initParams(inst :: QueryInstance, opts :: AdmmSdpOptions)
  ffnet = inst.ffnet
  input = inst.input

  γvardims = makeγvardims(opts.β, inst, opts.tband_func)
  γindim, γoutdim, γkdims = γvardims
  @assert length(γkdims) >= 2

  # Initialize the iteration variables
  γ = zeros(γindim + γoutdim + sum(γkdims))

  num_cliques = ffnet.K - opts.β - 1
  vdims = [size(Ec(k, opts.β, ffnet.zdims))[1] for k in 1:num_cliques]
  vs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  zs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  λs = [zeros(vdims[k]^2) for k in 1:num_cliques]
  params = AdmmParams(γ=γ, vs=vs, zs=zs, λs=λs, γindim=γindim, γoutdim=γoutdim, γkdims=γkdims)
  return params
end

# Cache precomputation
function precompute(inst :: QueryInstance, params :: AdmmParams, opts :: AdmmSdpOptions)
  @assert inst.ffnet.type isa ReluNetwork

  cache_start_time = time()

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
  Eout = [E1; EK; Ea]

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
  xinaff = vec(Ein' * makeXin(zeros(params.γindim), input, ffnet) * Ein)

  if inst isa SafetyInstance
    xoutaff = vec(Eout' * makeXout(inst.safety.S, ffnet) * Eout)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    S0 = makeShyperplane(inst.reach_set.normal, 0, ffnet)
    xoutaff = vec(Eout' * makeXout(S0, ffnet) * Eout)
  else
    error("unsupported instance: " * string(inst))
  end

  xkaffs = Vector{Vector{Float64}}()
  for k = 1:(num_cliques+1)
    EXk = [E(k, opts.β, zdims); Ea]
    qxdim = Qxdim(k, opts.β, zdims)
    xkaff = vec(EXk' * makeXqγ(k, opts.β, zeros(params.γkdims[k]), xqinfos[k]) * EXk)
    push!(xkaffs, xkaff)
  end

  zaff = xinaff + xoutaff + sum(xkaffs)

  # Now that we have computed the affine components, can begin actually computing J
  Jparts = Vector{Any}()

  # ... first computing Xin
  for i in 1:params.γindim
    xini = vec(Ein' * makeXin(e(i, params.γindim), input, ffnet) * Ein) - xinaff
    push!(Jparts, xini)
  end

  # ... then doing the Xout, but currently only applies if we're a reach instance
  if inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    S1 = makeShyperplane(inst.reach_set.normal, 1, ffnet)
    xout1 = vec(Eout' * makeXout(S1, ffnet) * Eout) - xoutaff
    push!(Jparts, xout1)
  end

  # ... and finally the Xks
  for k in 1:(num_cliques+1)
    EXk = [E(k, opts.β, zdims); Ea]
    for i in 1:params.γkdims[k]
      xki = vec(EXk' * makeXqγ(k, opts.β, e(i, params.γkdims[k]), xqinfos[k]) * EXk) - xkaffs[k]
      push!(Jparts, xki)
    end
  end

  # Now finish J
  J = hcat(Jparts...)

  # Some computation for the H matrices
  Hs = [kron(Ec(k, opts.β, zdims), Ec(k, opts.β, zdims)) for k in 1:num_cliques]

  cache_time = time() - cache_start_time

  cache = AdmmCache(J=J, zaff=zaff, Hs=Hs)
  return cache, cache_time
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
function stepXsolver(inst :: QueryInstance, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_PFEAS" => 1e-6,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  # Set up γ
  γdim = params.γindim + params.γoutdim + sum(params.γkdims)
  var_γ = @variable(model, [1:γdim])
  @constraint(model, var_γ .>= 0)

  # Set up vs
  num_cliques = params.num_cliques
  var_vs = Vector{Any}()
  for k in 1:num_cliques
    var_vk = @variable(model, [1:length(params.vs[k])])
    push!(var_vs, var_vk)
  end

  # The equality constraint
  Rhs = sum(cache.Hs[k]' * var_vs[k] for k in 1:num_cliques)
  @constraint(model, makezβ(params, cache, opts) .== Rhs)

  # The objective, but we need to pose it as a second order cone problem
  norms = @variable(model, [1:num_cliques])
  for k in 1:num_cliques
    termk = params.zs[k] - var_vs[k] + (params.λs[k] / opts.ρ)
    @constraint(model, [norms[k]; termk] in SecondOrderCone())
  end

  # When the safety instance, the objective is just the penalty term
  if inst isa SafetyInstance
    J = sum(norms[k]^2 for k in 1:num_cliques)
  # But when we are a hyperplane reachability set, also the γout is to be considered
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    # The γ[params.γindim+1] stores the γout
    J = var_γ[params.γindim+1] + sum(norms[k]^2 for k in 1:num_cliques)
  else
  end

  @objective(model, Min, J)

  # Solve and return
  optimize!(model)
  new_γ = value.(var_γ)
  new_vs = [value.(var_vs[k]) for k in 1:num_cliques]
  return new_γ, new_vs
end

#
function stepYsolver(inst :: QueryInstance, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_PFEAS" => 1e-6,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  num_cliques= params.num_cliques
  var_Zs = Vector{Any}()
  for k in 1:num_cliques
    zkdim = Int(round(sqrt(length(params.zs[k]))))
    var_Zk = @variable(model, [1:zkdim, 1:zkdim], Symmetric)
    @SDconstraint(model, var_Zk <= 0)
    push!(var_Zs, var_Zk)
  end

  # The norms
  norms = @variable(model, [1:num_cliques])
  for k in 1:num_cliques
    termk = vec(var_Zs[k]) - params.vs[k] + (params.λs[k] / opts.ρ)
    @constraint(model, [norms[k]; termk] in SecondOrderCone())
  end

  # The objective
  J = sum(norms[k]^2 for k in 1:num_cliques)

  # Solve and return
  optimize!(model)
  new_zs = [vec(value.(var_Zs[k])) for k in 1:num_cliques]
  return new_zs
end

#
function stepZ(inst :: QueryInstance, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_λs = [params.λs[k] + opts.ρ * (params.zs[k] - params.vs[k]) for k in 1:params.num_cliques]
  return new_λs
end

#
function isγSat(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  zβ = makezβ(params, cache, opts)
  zβdim = Int(round(sqrt(length(zβ))))
  Zβ = reshape(zβ, (zβdim, zβdim))
  if eigmax(Zβ) > opts.nsd_tol
    if opts.verbose; println("SAT!") end
    return true
  else
    return false
  end
end

#
function shouldStop(t :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  if t > opts.begin_check_at_iter && mod(t, opts.check_every_k_iters) == 0
    if isγSat(params, cache, opts); return true end
  end
  return false
end

#
function admm(inst :: QueryInstance, _params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  iters_run = 0
  total_time = 0
  iter_params = deepcopy(_params)

  for t = 1:opts.max_iters
    step_start_time = time()

    # X stuff
    x_start_time = time()
    # new_γ, new_vs = stepX(iter_params, cache, opts)
    new_γ, new_vs = stepXsolver(inst, iter_params, cache, opts)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    x_time = time() - x_start_time

    # Y stuff
    y_start_time = time()
    # new_zs = stepY(iter_params, cache, opts)
    new_zs = stepYsolver(inst, iter_params, cache, opts)
    iter_params.zs = new_zs
    y_time = time() - y_start_time

    # Z stuff
    z_start_time = time()
    new_λs = stepZ(inst, iter_params, cache, opts)
    iter_params.λs = new_λs
    z_time = time() - z_start_time

    # Primal residual
    # Coalesce time statistics
    step_time = time() - step_start_time
    total_time = total_time + step_time
    all_times = round.((x_time, y_time, z_time, step_time, total_time), digits=3)

    if opts.verbose
      println("step[" * string(t) * "/" * string(opts.max_iters) * "] times: " * string(all_times))
    end

    if shouldStop(t, iter_params, cache, opts); break end
  end

  return iter_params, total_time
end

# Call this
function run(inst :: SafetyInstance, opts :: AdmmSdpOptions)
  start_time = time()
  start_params = initParams(inst, opts)
  cache, setup_time = precompute(inst, start_params, opts)

  final_params, admm_time = admm(inst, start_params, cache, opts)
  total_time = time() - start_time
  
  status = isγSat(final_params, cache, opts) ? "OPTIMAL" : "SLOW_PROGRESS"


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

