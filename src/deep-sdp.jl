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
  use_xlims :: Bool = true
  use_slims :: Bool = true
  verbose :: Bool = false
end

# Calculate the xlims and slims, if applicable
function makeLimits(input :: InputConstraint, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  if input isa BoxInput
    xlimss, ylimss = propagateBox(input.x1min, input.x1max, ffnet)

    # Set up xlims
    xmin = vcat([xl[1] for xl in xlimss[2:end-1]]...)
    xmax = vcat([xl[2] for xl in xlimss[2:end-1]]...)
    xlims = (xmin, xmax)

    # Set up ymin and ymax first
    ymin = vcat([yl[1] for yl in ylimss]...)
    ymax = vcat([yl[2] for yl in ylimss]...)

    # Determine the Ipos and Ineg set in order to calculate slims
    ε = 1e-6
    Ipos = findall(z -> z > ε, ymin)
    Ineg = findall(z -> z < -ε, ymax)

    smin = zeros(length(ymin))
    smin[Ipos] .= 1.0

    smax = zeros(length(ymax))
    smax[Ineg] .= 0.0

    slims = (smin, smax)

    # Fiddle with options
    xlims = opts.use_xlims ? xlims : nothing
    slims = opts.use_slims ? slims : nothing

    return xlims, slims
  else
    @warn ("makeLimits: unsupported input " * string(input) * ", returning (nothing, nothing)")
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
  return MinP, γ
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
function makeMmidQ!(model, xlims, slims, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  qxdim = sum(ffnet.zdims[2:end-1])

  if ffnet.type isa ReluNetwork
    @variable(model, λ[1:qxdim] >= 0)
    @variable(model, τ[1:qxdim, 1:qxdim] >= 0, Symmetric)
    @variable(model, η[1:qxdim] >= 0)
    @variable(model, ν[1:qxdim] >= 0)
    @variable(model, d[1:qxdim] >= 0)

    b = ffnet.K - 1
    Qvars = (λ, τ, η, ν, d)
    MmidQ = makeXk(1, b, Qvars, ffnet, xlims=xlims, slims=slims)
    return MmidQ, Qvars
  else
    error("unsupported network: " * string(ffnet))
  end
end

# Solve the safety verification problem instance
function solveSafety(inst :: SafetyInstance, opts :: DeepSdpOptions)
  total_start_time = time()
  setup_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))
  
  # Make MinP and MoutS first because they're simpler
  MinP, γ = makeMinP!(model, inst.input, inst.ffnet, opts)
  MoutS = makeMoutS!(model, inst.safety.S, inst.ffnet, opts)
  
  # Construct limits and then MmidQ
  xlims, slims = makeLimits(inst.input, inst.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, xlims, slims, inst.ffnet, opts)

  # Now the LMI
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)

  # The time it took to set up the problem
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end

  # Now solve the problem
  optimize!(model)
  summary = solution_summary(model)
  if opts.verbose; println("solve time: " * string(summary.solve_time)) end

  # Prepare the output and return
  soln_dict = Dict(:γ => value.(γ), :Qvars => [value.(qv) for qv in Qvars])
  total_time = round(time() - total_start_time, digits=2)

  output = SolutionOutput(
            solution = soln_dict,
            summary = summary,
            status = string(summary.termination_status),
            total_time = total_time,
            setup_time = setup_time,
            solve_time = round(summary.solve_time, digits=2))
  return output
end

# Solve for the hyperplane offsets that bound the network output f(x)
function solveHyperplaneReachability(inst :: ReachabilityInstance, opts :: DeepSdpOptions)
  total_start_time = time()

  # Make the limits first
  xlims, slims = makeLimits(inst.input, inst.ffnet, opts)

  doffsets = Vector{Float64}()
  summaries = Vector{Any}()
  statuses = Vector{String}()
  soln_dicts = Vector{Any}()
  setup_times = Vector{Float64}()
  solve_times = Vector{Float64}()

  num_normals = length(inst.reach_set.normals)

  for (i, normal) in enumerate(inst.reach_set.normals)
    iter_setup_start_time = time()

    model = Model(optimizer_with_attributes(
      Mosek.Optimizer,
      "QUIET" => true,
      "INTPNT_CO_TOL_DFEAS" => 1e-6))

    # Make MinP first
    MinP, γ = makeMinP!(model, inst.input, inst.ffnet, opts)

    # Make MmidQ next
    MmidQ, Qvars = makeMmidQ!(model, xlims, slims, inst.ffnet, opts)

    # Now set up MoutS
    @variable(model, doffset)
    S = makeShyperplane(normal, doffset, inst.ffnet)
    MoutS = makeMoutS!(model, S, inst.ffnet, opts)

    # Now set up the LMi and objective
    Z = MinP + MmidQ + MoutS
    @SDconstraint(model, Z <= 0)
    @objective(model, Min, doffset)

    # Calculate setup times
    iter_setup_time = time() - iter_setup_start_time
    push!(setup_times, iter_setup_time)

    # Now solve the problem
    optimize!(model)
    summary = solution_summary(model)
    push!(summaries, summary)
    push!(statuses, string(summary.termination_status))

    iter_solve_time = round.(summary.solve_time, digits=2)
    push!(solve_times, iter_solve_time)
    if opts.verbose; println("iter[" * string(i) * "/" * string(num_normals) * "] solve time: " * string(iter_solve_time)) end

    # Store results
    push!(doffsets, value(doffset))

    soln_dict = Dict(:γ => value.(γ), :Qvars => [value.(qv) for qv in Qvars])
    push!(soln_dicts, soln_dict)
  end

  # Prepare and return the output
  cds = [(c,d) for (c,d) in zip(inst.reach_set.normals, doffsets)]
  soln_dict = Dict(:cds => cds, :iter_soln_dicts => soln_dicts)

  total_setup_time = round(sum(setup_times), digits=2)
  total_solve_time = round(sum(solve_times), digits=2)
  total_time = round(time() - total_start_time, digits=2)

  output = SolutionOutput(
            solution = soln_dict,
            summary = summaries,
            status = statuses,
            total_time = total_time,
            setup_time = total_setup_time,
            solve_time = total_solve_time)
  return output
end

# The interface to call
function run(inst :: QueryInstance, opts :: DeepSdpOptions)
  if inst isa SafetyInstance
    return solveSafety(inst, opts)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    return solveHyperplaneReachability(inst, opts)
  else
    error("unrecognized query instance: " * string(inst))
  end
end

# For debugging purposes we export more than what is needed

# export SetupMethod, BigSetup
export DeepSdpOptions
export run

end # End module

