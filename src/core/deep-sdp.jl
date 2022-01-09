# Implementation of the DeepSdp algorithm
module DeepSdp

using ..Header
using ..Common
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Configuration options
@with_kw struct DeepSdpOptions
  x_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  tband :: Union{Nothing, Int} = nothing
  verbose :: Bool = false
end

# Computes the MinP matrix. Treat this function as though it modifies the model
function makeMinP!(model, input :: InputConstraint, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  if input isa BoxInput
    @variable(model, γ[1:ffnet.xdims[1]] >= 0)
  elseif input isa PolytopeInput
    @variable(model, γ[1:ffnet.xdims[1]^2] >= 0)
  else
    error("unsupported input constraints: " * string(input))
  end
  E1 = E(1, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Ein = [E1; Ea]
  Xin = makeXin(γ, input, ffnet)
  MinP = Ein' * Xin * Ein
  Pvars = Dict(:γ => γ)
  return MinP, Pvars
end

# Computes the MoutS matrix. Treat this function as though it modifies the model
function makeMoutS!(model, S, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Esafe = [E1; EK; Ea]
  Xsafe = makeXsafe(S, ffnet)
  MoutS = Esafe' * Xsafe * Esafe
  return MoutS
end

# Make the MmidQ matrix. Treat this function as though it modifies the model
function makeMmidQ!(model, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  β = ffnet.K - 1
  qxdim = Qxdim(1, β, ffnet.zdims)
  if ffnet.type isa ReluNetwork

    # The actual tband must be in range 0 <= tband <= qxdim - 1, so adjust accordingly
    tband = (opts.tband isa Nothing) ? (qxdim-1) : max(0, min(opts.tband, qxdim-1))
    if opts.verbose && !(opts.tband isa Nothing) && (tband != opts.tband)
      @warn ("adjusted opts.tband from " * string(opts.tband) * " to " * string(tband))
    end

    λ_slope_length = λlength(qxdim, tband)
    @variable(model, λ_slope[1:λ_slope_length] >= 0)
    @variable(model, η_slope[1:qxdim] >= 0)
    @variable(model, ν_slope[1:qxdim] >= 0)
    @variable(model, d_out[1:qxdim] >= 0)

    vars = (λ_slope, η_slope, ν_slope, d_out)
    xqinfo = Xqinfo(
      ffnet = ffnet,
      ϕout_intv = selectϕoutIntervals(1, β, opts.x_intvs),
      slope_intv = selectSlopeIntervals(1, β, opts.slope_intvs),
      tband = tband)

    MmidQ = makeXq(1, β, vars, xqinfo)
    Qvars = Dict(:λ_slope => λ_slope,
                  :η_slope => η_slope,
                  :ν_slope => ν_slope,
                  :d_out => d_out)
    return MmidQ, Qvars
  else
    error("unsupported network: " * string(ffnet))
  end
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, inst :: SafetyInstance, opts :: DeepSdpOptions)
  setup_start_time = time()

  # Make the components
  MinP, Pvars = makeMinP!(model, inst.input, inst.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, inst.ffnet, opts)
  MoutS = makeMoutS!(model, inst.safety.S, inst.ffnet, opts)

  # Now the LMI
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)

  # The time it took to set up the problem
  vars = merge(Pvars, Qvars)
  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Set up the model for hyperplane reachability (optimality)
function setupHyperplaneReachability!(model, inst :: ReachabilityInstance, opts :: DeepSdpOptions)
  @assert inst.reach_set isa HyperplaneSet
  setup_start_time = time()

  # Make MinP and MmidQ first
  MinP, Pvars = makeMinP!(model, inst.input, inst.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, inst.ffnet, opts)

  # Now set up MoutS
  @variable(model, h)
  Svars = Dict(:h => h)
  S = makeShyperplane(inst.reach_set.normal, h, inst.ffnet)
  MoutS = makeMoutS!(model, S, inst.ffnet, opts)

  # Now set up the LMI and objective
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)
  @objective(model, Min, h)

  # Calculate setup times and return
  vars = merge(Pvars, Qvars, Svars)
  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts :: DeepSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = round(summary.solve_time, digits=3)
  if opts.verbose; println("solve time: " * string(solve_time)) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, solve_time
end

# The interface to call
function run(inst :: QueryInstance, opts :: DeepSdpOptions)
  total_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9))

  # Delegate the appropriate call depending on our query instance
  if inst isa SafetyInstance
    _, vars, setup_time = setupSafety!(model, inst, opts)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    _, vars, setup_time = setupHyperplaneReachability!(model, inst, opts)
  else
    error("unrecognized query instance: " * string(inst))
  end

  # Get ready to return
  summary, values, solve_time = solve!(model, vars, opts)
  total_time = round(time() - total_start_time, digits=3)
  if opts.verbose; println("total time: " * string(total_time)) end
  return SolutionOutput(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = summary.termination_status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = solve_time)
end

# export SetupMethod, BigSetup
export DeepSdpOptions
export run

end # End module

