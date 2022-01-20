module AdmmSdp

using ..Header
using ..Common
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using Mosek
using MosekTools
using Printf


# The options for ADMM
@with_kw struct AdmmSdpOptions
  max_iters :: Int = 200
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  cholesky_reg_ε :: Float64 = 1e-2

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
  Hs_hots :: Vector{Vector{Int}}

  # The cholesky factorization of (D + ρJJ') + εI
  chol :: Cholesky{Float64, Matrix{Float64}}

  # The entries corresponding to the regularization term
  diagL_zeros :: BitArray{1}
end

# Status of ADMM
abstract type AdmmStatus end
struct StillRunning <: AdmmStatus end
struct FoundγSat <: AdmmStatus end
struct MaxIters <: AdmmStatus end
@with_kw struct SmallError <: AdmmStatus
  ε :: Float64
end

# Admm summary
@with_kw struct AdmmSummary
  steps_taken :: Int
  termination_status :: AdmmStatus
  stepping_time :: Float64
  total_X_time :: Float64
  total_Y_time :: Float64
  total_Z_time :: Float64
  avg_X_time :: Float64
  avg_Y_time :: Float64
  avg_Z_time :: Float64
end

# Initialize zero-valued parameters of the appropriate size
function initParams(inst :: SafetyInstance, opts :: AdmmSdpOptions)
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

# Supposed to do a fast Ec' * X * Ec computation.
# Takes a 2-block square matrix X and splats it into a square Z matrix
# Assumes that:
# size(X11) == (d1, d1), size(X12) == (d1, d2), size(X21) == (d2, d1), size(X22) == (d2, d2)
# i1 and i2 denote the respective insertion indices in Z
function _expand2(X, (d1, i1), (d2, i2), Zdim :: Int)
  @assert d1 >= 1 && d2 >= 1   # Sane dimensions
  @assert i1 >= 1 && i2 >= 2   # Sane insertions
  @assert i1 + d1 <= i2        # X11 will be inserted before X12
  @assert i2 + d2 - 1 <= Zdim  # Don't overrun the end

  Xdim = size(X)[1]
  Z = zeros(Zdim, Zdim)
  Z[i1:(i1+d1-1), i1:(i1+d1-1)] = view(X, 1:d1, 1:d1)
  Z[i1:(i1+d1-1), i2:(i2+d2-1)] = view(X, 1:d1, (d1+1):Xdim)
  Z[i2:(i2+d2-1), i1:(i1+d1-1)] = view(X, (d1+1):Xdim, 1:d1)
  Z[i2:(i2+d2-1), i2:(i2+d2-1)] = view(X, (d1+1):Xdim, (d1+1):Xdim)
  return Z
end

function _fastEintXEin(X, zdims :: Vector{Int}, Zdim :: Int)
  return _expand2(X, (zdims[1], 1), (1, Zdim), Zdim)
end

function _fastEouttXEout(X, zdims :: Vector{Int}, Zdim :: Int)
  d2 = zdims[end-1] + 1
  return _expand2(X, (zdims[1], 1), (d2, Zdim-d2+1), Zdim)
end

function _fastEXktXEXk(k :: Int, X, zdims :: Vector{Int}, Zdim :: Int)
  d = size(X)[1]
  d2 = 1
  d1 = d - d2
  i1 = sum(zdims[1:k-1]) + 1
  return _expand2(X, (d1, i1), (d2, Zdim-d2+1), Zdim)
end

function _fastH(Hs_hot, x)
  return x[Hs_hot == 1]
end

# Cache precomputation
function precompute(inst :: SafetyInstance, params :: AdmmParams, opts :: AdmmSdpOptions)
  @assert inst.ffnet.type isa ReluNetwork

  cache_start_time = time()

  input = inst.input
  ffnet = inst.ffnet
  zdims = ffnet.zdims
  Zdim = sum(zdims)
  γdim = params.γindim + sum(params.γkdims)

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
  xinaff = vec(_fastEintXEin(makeXin(zeros(params.γindim), input, ffnet), zdims, Zdim))
  xoutaff = vec(_fastEouttXEout(makeXout(inst.safety.S, ffnet), zdims, Zdim))

  xkaffs = Vector{Vector{Float64}}()
  for k = 1:(num_cliques+1)
    xkaff = vec(_fastEXktXEXk(k, makeXqγ(k, opts.β, zeros(params.γkdims[k]), xqinfos[k]), zdims, Zdim))
    push!(xkaffs, xkaff)
  end

  zaff = xinaff + xoutaff + sum(xkaffs)

  # Now that we have computed the affine components, can begin actually computing J
  J = zeros(Zdim^2, γdim)
  next_J_col = 1

  # ... first computing Xin
  for i in 1:params.γindim
    xini = vec(_fastEintXEin(makeXin(e(i, params.γindim), input, ffnet), zdims, Zdim)) - xinaff
    J[:,next_J_col] = xini
    next_J_col += 1
  end

  # Now do the other Xks
  for k in 1:(num_cliques+1)
    for i in 1:params.γkdims[k]
      xki = vec(_fastEXktXEXk(k, makeXqγ(k, opts.β, e(i, params.γkdims[k]), xqinfos[k]), zdims, Zdim)) - xkaffs[k]
      J[:,next_J_col] = xki
      next_J_col += 1
    end
  end

  # Some computation for the H matrices
  Hs_start_time = time()
  Hs = [kron(Ec(k, opts.β, zdims), Ec(k, opts.β, zdims)) for k in 1:num_cliques]
  Hs_hots = [Int.(Hs[k]' * ones(size(Hs[k])[1])) for k in 1:params.num_cliques]
  @printf("\tHs time: %.3f\n", Hs_start_time)


  DJJt_start_time = time()
  D = Diagonal(sum(Hs_hots))
  DJJt = D + (opts.ρ * J * J')
  DJJt_reg = Symmetric(DJJt + opts.cholesky_reg_ε * I)
  @printf("\tDJJt time: %.3f\n", time() - DJJt_start_time)

  chol_start_time = time()
  chol = cholesky(DJJt_reg)
  @printf("\tcholesky time: %.3f\n", time() - chol_start_time)

  diagL_start_time = time()
  diagL_zeros = (diag(chol.L) .<= 2 * sqrt(opts.cholesky_reg_ε))
  @printf("\tdiagL time: %.3f\n", time() - diagL_start_time)

  cache_time = time() - cache_start_time
  cache = AdmmCache(J=J, zaff=zaff, Hs=Hs, Hs_hots=Hs_hots, chol=chol, diagL_zeros=diagL_zeros)
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
function projectNsd(xk :: Vector{Float64})
  dim = Int(round(sqrt(length(xk)))) # :)
  @assert length(xk) == dim * dim
  tmp = Symmetric(reshape(xk, (dim, dim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

#
function stepXsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
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
  
  γnorm = @variable(model)
  @constraint(model, [γnorm; var_γ] in SecondOrderCone())

  obj = γnorm^2 + sum(norms[k]^2 for k in 1:num_cliques)
  @objective(model, Min, obj)

  # Solve and return
  optimize!(model)
  new_γ = value.(var_γ)
  new_vs = [value.(var_vs[k]) for k in 1:num_cliques]
  return new_γ, new_vs
end

#
function stepX(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  num_cliques = params.num_cliques
  b = -cache.zaff + sum(cache.Hs[k]' * (params.zs[k] + (params.λs[k] / opts.ρ)) for k in 1:num_cliques)

  tmp = b
  tmp[cache.diagL_zeros] .= 0
  tmp = cache.chol.L \ tmp
  tmp[cache.diagL_zeros] .= 0
  tmp = cache.chol.L' \ tmp
  tmp[cache.diagL_zeros] .= 0
  new_x = tmp

  # new_x = -cache.pinv_D_ρJJt * b
  new_γ = projectNonnegative(-opts.ρ * cache.J' * new_x)
  new_vs = [params.zs[k] + (params.λs[k] / opts.ρ) + cache.Hs[k] * new_x for k in 1:num_cliques]
  return new_γ, new_vs
end

#
function stepYsolver(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
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
function stepY(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  tmps = [params.vs[k] - (params.λs[k] / opts.ρ) for k in 1:params.num_cliques]
  new_zs = projectNsd.(tmps)
  return new_zs
end

#
function stepZ(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  new_λs = [params.λs[k] + opts.ρ * (params.zs[k] - params.vs[k]) for k in 1:params.num_cliques]
  return new_λs
end

#
function isγSat(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  zβ = makezβ(params, cache, opts)
  zβdim = Int(round(sqrt(length(zβ))))
  Zβ = reshape(zβ, (zβdim, zβdim))
  if eigmax(Zβ) > opts.nsd_tol
    if opts.verbose; @printf("SAT!\n") end
    return true
  else
    return false
  end
end

function checkStatus(t :: Int, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  if t > opts.begin_check_at_iter && mod(t, opts.check_every_k_iters) == 0
    if isγSat(params, cache, opts); return FoundγSat() end
  end
  return StillRunning()
end

#
function admm(_params :: AdmmParams, cache :: AdmmCache, opts :: AdmmSdpOptions)
  iter_params = deepcopy(_params)

  steps_taken = 0
  term_status = MaxIters()
  total_step_time = 0
  total_X_time = 0.0
  total_Y_time = 0.0
  total_Z_time = 0.0

  for t = 1:opts.max_iters
    step_start_time = time()

    # X stuff
    x_start_time = time()
    new_γ, new_vs = stepX(iter_params, cache, opts)
    # new_γ, new_vs = stepXsolver(iter_params, cache, opts)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    X_time = time() - x_start_time
    total_X_time += X_time

    # Y stuff
    y_start_time = time()
    new_zs = stepY(iter_params, cache, opts)
    # new_zs = stepYsolver(iter_params, cache, opts)
    iter_params.zs = new_zs
    Y_time = time() - y_start_time
    total_Y_time += Y_time 

    # Z stuff
    z_start_time = time()
    new_λs = stepZ(iter_params, cache, opts)
    iter_params.λs = new_λs
    Z_time = time() - z_start_time
    total_Z_time += Z_time

    # Primal residual
    # Coalesce time statistics
    step_time = time() - step_start_time
    total_step_time = total_step_time + step_time
    times_str = @sprintf("(%.3f, %.3f, %.3f, %.3f, %.3f)", X_time, Y_time, Z_time, step_time, total_step_time)

    if opts.verbose
        @printf("\tstep[%d/%d] times: %s\n", t, opts.max_iters, times_str)
    end

    steps_taken += 1

    status = checkStatus(t, iter_params, cache, opts)
    if status isa FoundγSat
      term_status = status
      break
    end
  end

  summary = AdmmSummary(
    steps_taken = steps_taken,
    termination_status = term_status,
    stepping_time = total_step_time,
    total_X_time = total_X_time,
    total_Y_time = total_Y_time,
    total_Z_time = total_Z_time,
    avg_X_time = total_X_time / steps_taken,
    avg_Y_time = total_Y_time / steps_taken,
    avg_Z_time = total_Z_time / steps_taken)
  return iter_params, summary
end

# Call this
function run(inst :: SafetyInstance, opts :: AdmmSdpOptions)
  start_time = time()
  start_params = initParams(inst, opts)
  cache, setup_time = precompute(inst, start_params, opts)

  final_params, summary = admm(start_params, cache, opts)
  total_time = time() - start_time
  
  return SolutionOutput(
    objective_value = 0.0,
    values = final_params,
    summary = summary,
    termination_status = summary.termination_status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.stepping_time)
end

export initParams, precomputeCache
export AdmmSdpOptions, AdmmParams, AdmmCache

end # End module

