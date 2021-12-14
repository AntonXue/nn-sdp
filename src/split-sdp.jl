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
  β :: Int = 1 # @assert 1 <= β <= ffnet.K - 2
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
  return Xinit, γ
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

    Qkvars = (λ, τ, η, ν, d)
    Xk = makeXk(k, opts.β, Qkvars, ffnet, xlims=xcklims, slims=scklims)
    return Xk, Qkvars
  else
    error("unsupported network: " * string(ffnet))
  end
end

# Solve the safety problem
function solveSafety(inst :: SafetyInstance, opts :: SplitSdpOptions)
  total_start_time = time()
  setup_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  # Interval propagation
  xlimTups, slimTups = findLimitTuples(inst, opts)

  E1 = E(1, inst.ffnet.zdims)
  EK = E(inst.ffnet.K, inst.ffnet.zdims)
  Ea = E(inst.ffnet.K+1, inst.ffnet.zdims)

  Einit = [E1; Ea]
  Xinit, γ = makeSplitXinit!(model, inst.input, inst.ffnet, opts)

  Esafe = [E1; EK; Ea]
  Xsafe = makeSplitXsafe!(model, inst.safety.S, inst.ffnet, opts)

  Z = (Einit' * Xinit * Einit) + (Esafe' * Xsafe * Esafe)

  allQvars = Vector{Any}()
  numXks = inst.ffnet.K - opts.β
  for k in 1:numXks
    if opts.verbose; println("making X[" * string(k) * "/" * string(numXks) * "]") end
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    xcklims, scklims = selectLimits(k, opts.β, xlimTups, slimTups)
    Xk, Qkvars = makeSplitXk!(model, k, xcklims, scklims, inst.ffnet, opts)
    Z = Z + (EXk' * Xk * EXk)
    push!(allQvars, Qkvars)
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
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end

  # Now solve the problem
  optimize!(model)
  summary = solution_summary(model)
  if opts.verbose; println("solve time: " * string(summary.solve_time)) end

  # Prepare the output and return
  allQvalues = [[value.(qkv) for qkv in Qkvars] for Qkvars in allQvars]
  value_dict = Dict(:γ => value.(γ), :allQvars => allQvalues)
  total_time = round(time() - total_start_time, digits=2)

  return SolutionOutput(
    values = value_dict,
    summary = summary,
    status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = round(summary.solve_time, digits=2))
end

#
function solveHyperplaneReachability(inst :: ReachabilityInstance, opts :: SplitSdpOptions)
  total_start_time = time()

  # Interval propagation
  xlimTups, slimTups = findLimitTuples(inst, opts)

  # Set up useful matrices
  E1 = E(1, inst.ffnet.zdims)
  EK = E(inst.ffnet.K, inst.ffnet.zdims)
  Ea = E(inst.ffnet.K+1, inst.ffnet.zdims)
  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)

  #
  doffsets = Vector{Float64}()
  summaries = Vector{Any}()
  statuses = Vector{String}()
  value_dicts = Vector{Any}()
  setup_times = Vector{Float64}()
  solve_times = Vector{Float64}()

  numNormals = length(inst.reach_set.normals)
  for (i, normal) in enumerate(inst.reach_set.normals)
    iter_setup_start_time = time()
    model = Model(optimizer_with_attributes(
      Mosek.Optimizer,
      "QUIET" => true,
      "INTPNT_CO_TOL_DFEAS" => 1e-6))

    # Make Xinit
    Xinit, γ = makeSplitXinit!(model, inst.input, inst.ffnet, opts)
    Z = (Einit' * Xinit * Einit)

    # Make the Xks
    allQvars = Vector{Any}()
    numXks = inst.ffnet.K - opts.β
    for k in 1:numXks
      if opts.verbose; println("making X[" * string(k) * "/" * string(numXks) * "]") end
      Ekβ = E(k, opts.β, inst.ffnet.zdims)
      EXk = [Ekβ; Ea]
      xcklims, scklims = selectLimits(k, opts.β, xlimTups, slimTups)
      Xk, Qkvars = makeSplitXk!(model, k, xcklims, scklims, inst.ffnet, opts)
      Z = Z + (EXk' * Xk * EXk)
      push!(allQvars, Qkvars)
    end

    # Now set up Xsafe
    @variable(model, doffset)
    S = makeShyperplane(normal, doffset, inst.ffnet)
    Xsafe = makeSplitXsafe!(model, S, inst.ffnet, opts)

    # Fully complete Z
    Z = Z + (Esafe' * Xsafe * Esafe)

    # Select and assert that each Zk is NSD
    scaledZ = Z .* Ωinv
    numCliques = inst.ffnet.K - opts.β - 1
    for k = 1:numCliques
      Eck = Ec(k, opts.β, inst.ffnet.zdims)
      Zk = Eck * scaledZ * Eck'
      @SDconstraint(model, Zk <= 0)
    end

    # Objective
    @objective(model, Min, doffset)

    # Calculate the setup times
    iter_setup_time = time() - iter_setup_start_time

    # Now solve the problem
    optimize!(model)
    summary = solution_summary(model)

    iter_solve_time = round.(summary.solve_time, digits=2)
    if opts.verbose; println("reach iter[" * string(i) * "/" * string(numNormals) * "] solve time: " * string(iter_solve_time)) end

    # Store results
    push!(doffsets, value(doffset))
    push!(summaries, summary)
    push!(statuses, string(summary.termination_status))
    push!(setup_times, iter_setup_time)
    push!(solve_times, iter_solve_time)

    allQvalues = [[value.(qkv) for qkv in Qkvars] for Qkvars in allQvars]
    value_dict = Dict(:γ => value.(γ), :allQvalues => allQvalues)
    push!(value_dicts, value_dict)
  end

  # Prepare and return the output
  cds = [(c,d) for (c,d) in zip(inst.reach_set.normals, doffsets)]
  value_dict = Dict(:cds => cds, :iter_value_dicts => value_dicts)

  total_setup_time = round(sum(setup_times), digits=2)
  total_solve_time = round(sum(solve_times), digits=2)
  total_time = round(time() - total_start_time, digits=2)

  return SolutionOutput(
    values = value_dict,
    summary = summaries,
    status = statuses,
    total_time = total_time,
    setup_time = total_setup_time,
    solve_time = total_solve_time)
end

##

# Depending on what kind of setup we want to do

# The interface to call
function run(inst :: QueryInstance, opts :: SplitSdpOptions)

  if inst isa SafetyInstance
    return solveSafety(inst, opts)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    return solveHyperplaneReachability(inst, opts)
  else
    error("unrecognized query instance: " * string(inst))
  end
end

# For debugging purposes we export more than what is needed
export SumXThenSplitSetup
export SplitSdpOptions
export setup, solve!, run

end # End module

