# Implementation of the chordal decomposition of DeepSdp
module SplitSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Options
@with_kw struct SplitSdpOptions
  β :: Int = 1
  use_xintervals :: Bool = true
  use_localized_slopes :: Bool = true
  verbose :: Bool = false
end

# Calculate the interval propagation
function findLimitTuples(inst, opts :: SplitSdpOptions)
  if !hasproperty(inst, :input); return nothing, nothing end

  if inst.input isa BoxInput && (opts.use_xintervals || opts.use_localized_slopes)
    xlimTups, _, slimTups = propagateBox(inst.input.x1min, inst.input.x1max, inst.ffnet)
    xlimTups = opts.use_xintervals ? xlimTups : nothing
    slimTups = opts.use_localized_slopes ? slimTups : nothing
    return xlimTups, slimTups
  else
    return nothing, nothing
  end
end

# Find the pair of limits for (x[k+1], s[k]), ..., (x[k+b], s[k+b-1])
# This is because Ck has {x[k], x[k+1], ..., x[k+β]},
# so the outputs are x[k+1] ... x[k+β] and the inputs are x[k] ... x[k+β-1]
function selectLimits(k :: Int, β :: Int, xlimTups, slimTups)
  if xlimTups isa Nothing
    xlims = nothing
  else
    xmin = vcat([xl[1] for xl in xlimTups[k+1:k+β]]...)
    xmax = vcat([xl[2] for xl in xlimTups[k+1:k+β]]...)
    xlims = (xmin, xmax)
  end

  if slimTups isa Nothing
    slims = nothing
  else
    smin = vcat([sl[1] for sl in slimTups[k:k+β-1]]...)
    smax = vcat([sl[2] for sl in slimTups[k:k+β-1]]...)
    slims = (smin, smax)
  end

  return xlims, slims
end

# Treat this as though it modifies the model
function makeSplitXinit!(model, input :: InputConstraint, ffnet :: FeedForwardNetwork, opts :: SplitSdpOptions)
  if input isa BoxInput
    @variable(model, γ[1:ffnet.xdims[1]] >= 0)
  elseif input isa PolytopeInput
    @variable(model, γ[1:ffnet.xdims[1]^2] >= 0)
  else
    error("unsupported input constraints: " * string(input))
  end
  Xinit = makeXinit(γ, input, ffnet)
  return Xinit, Dict(:γ => γ)
end

# Treat this as though it modifies the model
function makeSplitXsafe!(model, S, ffnet :: FeedForwardNetwork, opts :: SplitSdpOptions)
  Xsafe = makeXsafe(S, ffnet)
  return Xsafe
end

# Treat this as though it modifies the model
function makeSplitXk!(model, k :: Int, xcklims, scklims, ffnet :: FeedForwardNetwork, opts :: SplitSdpOptions)
  qxdim = sum(ffnet.zdims[k+1:k+opts.β])
  if ffnet.type isa ReluNetwork
    λ = @variable(model, [1:qxdim])
    τ = @variable(model, [1:qxdim, 1:qxdim], Symmetric)
    η = @variable(model, [1:qxdim])
    ν = @variable(model, [1:qxdim])
    d = @variable(model, [1:qxdim])

    @constraint(model, λ[1:qxdim] .>= 0)
    @constraint(model, τ[1:qxdim, 1:qxdim] .>= 0)
    @constraint(model, η[1:qxdim] .>= 0)
    @constraint(model, ν[1:qxdim] .>= 0)
    @constraint(model, d[1:qxdim] .>= 0)

    vars = (λ, τ, η, ν, d)
    Xk = makeXk(k, opts.β, vars, ffnet, xlims=xcklims, slims=scklims)
    symk(v :: String) = Symbol(v * string(k))
    return Xk, Dict(symk("λ") => λ, symk("τ") => τ, symk("η") => η, symk("ν") => ν, symk("d") => d)
  else
    error("unsupported network: " * string(ffnet))
  end
end

#
function setupSafety!(model, inst :: SafetyInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

  # Interval propagation
  xlimTups, slimTups = findLimitTuples(inst, opts)

  # Some helpful block index matrices
  E1 = E(1, inst.ffnet.zdims)
  EK = E(inst.ffnet.K, inst.ffnet.zdims)
  Ea = E(inst.ffnet.K+1, inst.ffnet.zdims)
  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Setup Xinit and Xsafe first
  Xinit, Pvars = makeSplitXinit!(model, inst.input, inst.ffnet, opts)
  Xsafe = makeSplitXsafe!(model, inst.safety.S, inst.ffnet, opts)
  Z = (Einit' * Xinit * Einit) + (Esafe' * Xsafe * Esafe)

  # Make each Zk and add them to Z
  Qvars = Dict()
  numXks = inst.ffnet.K - opts.β
  for k in 1:numXks
    if opts.verbose; println("making X[" * string(k) * "/" * string(numXks) * "]") end
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    xcklims, scklims = selectLimits(k, opts.β, xlimTups, slimTups)
    Xk, Qkvars = makeSplitXk!(model, k, xcklims, scklims, inst.ffnet, opts)
    Z = Z + (EXk' * Xk * EXk)
    merge!(Qvars, Qkvars)
  end

  # Select and assert that each Zk is NSD
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)
  scaledZ = Z .* Ωinv
  numCliques = inst.ffnet.K - opts.β - 1
  for k = 1:numCliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * scaledZ * Eck'
    @SDconstraint(model, Zk <= 0)
  end

  # Time it took to set up the problem
  vars = merge(Pvars, Qvars)
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

#
function setupHyperplaneReachability!(model, inst :: ReachabilityInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

  # Interval propagation
  xlimTups, slimTups = findLimitTuples(inst, opts)

  # Some helpful block index matrices
  E1 = E(1, inst.ffnet.zdims)
  EK = E(inst.ffnet.K, inst.ffnet.zdims)
  Ea = E(inst.ffnet.K+1, inst.ffnet.zdims)
  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Setup Xinit first
  Xinit, Pvars = makeSplitXinit!(model, inst.input, inst.ffnet, opts)
  Z = Einit' * Xinit * Einit

  # Setup Xsafe and add it to Z
  @variable(model, h)
  Svars = Dict(:h => h)
  S = makeShyperplane(inst.reach_set.normal, h, inst.ffnet)
  Xsafe = makeSplitXsafe!(model, S, inst.ffnet, opts)
  Z = Z + (Esafe' * Xsafe * Esafe)

  # Make each Zk and add them to Z
  Qvars = Dict()
  numXks = inst.ffnet.K - opts.β
  for k in 1:numXks
    if opts.verbose; println("making X[" * string(k) * "/" * string(numXks) * "]") end
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    xcklims, scklims = selectLimits(k, opts.β, xlimTups, slimTups)
    Xk, Qkvars = makeSplitXk!(model, k, xcklims, scklims, inst.ffnet, opts)
    Z = Z + (EXk' * Xk * EXk)
    merge!(Qvars, Qkvars)
  end

  # Select and assert that each Zk is NSD
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)
  scaledZ = Z .* Ωinv
  numCliques = inst.ffnet.K - opts.β - 1
  for k = 1:numCliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * scaledZ * Eck'
    @SDconstraint(model, Zk <= 0)
  end

  # And the objective
  @objective(model, Min, h)

  # Time it took to set up the problem
  vars = merge(Pvars, Qvars, Svars)
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts :: SplitSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = round(summary.solve_time, digits=2)
  if opts.verbose; println("solve time: " * string(solve_time)) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, solve_time
end

# The interface to call
function run(inst :: QueryInstance, opts :: SplitSdpOptions)
  @assert 1 <= opts.β <= inst.ffnet.K - 2
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

# For debugging purposes we export more than what is needed
export SumXThenSplitSetup
export SplitSdpOptions
export setup, solve!, run

end # End module

