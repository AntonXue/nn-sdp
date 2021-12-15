# Implementation of the DeepSdp algorithm
module DeepSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Configuration options
@with_kw struct DeepSdpOptions
  use_xintervals :: Bool = true
  use_localized_slopes :: Bool = true
  verbose :: Bool = false
end

# Find pairs the limits of form (x[k+1], s[k])
function findLimits(inst, opts :: DeepSdpOptions)
  if !hasproperty(inst, :input); return nothing, nothing end

  if inst.input isa BoxInput && (opts.use_xintervals || opts.use_localized_slopes)
    xlimTups, _, slimTups = propagateBox(inst.input.x1min, inst.input.x1max, inst.ffnet)
    xmin = vcat([xl[1] for xl in xlimTups[2:end-1]]...) # Only care about the intermediates
    xmax = vcat([xl[2] for xl in xlimTups[2:end-1]]...)
    xlims = opts.use_xintervals ? (xmin, xmax) : nothing

    smin = vcat([sl[1] for sl in slimTups]...)
    smax = vcat([sl[2] for sl in slimTups]...)
    slims = opts.use_localized_slopes ? (smin, smax) : nothing
    return xlims, slims
  else
    # @warn "Propagation bounds available for BoxInput only"
    return nothing, nothing
  end
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
  Einit = [E1; Ea]
  Xinit = makeXinit(γ, input, ffnet)
  MinP = Einit' * Xinit * Einit
  return MinP, Dict(:γ => γ)
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
function makeMmidQ!(model, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions; xlims = nothing, slims = nothing)
  qxdim = sum(ffnet.zdims[2:end-1])
  if ffnet.type isa ReluNetwork
    @variable(model, λ[1:qxdim] >= 0)
    @variable(model, τ[1:qxdim, 1:qxdim] >= 0, Symmetric)
    @variable(model, η[1:qxdim] >= 0)
    @variable(model, ν[1:qxdim] >= 0)
    @variable(model, d[1:qxdim] >= 0)
    β = ffnet.K - 1
    vars = (λ, τ, η, ν, d)
    MmidQ = makeXk(1, β, vars, ffnet, xlims=xlims, slims=slims)
    return MmidQ, Dict(:λ => λ, :τ => τ, :η => η, :ν => ν, :d => d)
  else
    error("unsupported network: " * string(ffnet))
  end
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, inst :: SafetyInstance, opts :: DeepSdpOptions)
  setup_start_time = time()

  # Interval propagation
  xlims, slims = findLimits(inst, opts)

  # Make the components
  MinP, Pvars = makeMinP!(model, inst.input, inst.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, inst.ffnet, opts, xlims=xlims, slims=slims)
  MoutS = makeMoutS!(model, inst.safety.S, inst.ffnet, opts)

  # Now the LMI
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)

  # The time it took to set up the problem
  vars = merge(Pvars, Qvars)
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Set up the model for hyperplane reachability (optimality)
function setupHyperplaneReachability!(model, inst :: ReachabilityInstance, opts :: DeepSdpOptions)
  @assert inst.reach_set isa HyperplaneSet
  setup_start_time = time()

  # Interval propagation
  xlims, slims = findLimits(inst, opts)

  # Make MinP and MmidQ first
  MinP, Pvars = makeMinP!(model, inst.input, inst.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, inst.ffnet, opts, xlims=xlims, slims=slims)

  # Now set up MoutS
  @variable(model, h)
  S = makeShyperplane(inst.reach_set.normal, h, inst.ffnet)
  MoutS = makeMoutS!(model, S, inst.ffnet, opts)

  # Now set up the LMI and objective
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)
  @objective(model, Min, h)

  # Calculate setup times and return
  vars = merge(Dict(:h => h), Pvars, Qvars)
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts :: DeepSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = round(summary.solve_time, digits=2)
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
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

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
  total_time = round(time() - total_start_time, digits=2)
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

