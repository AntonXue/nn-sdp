# Implementation of the chordal decomposition of DeepSdp
module SplitSdp

using ..Header
using ..Common
# using ..Partitions
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Options
@with_kw struct SplitSdpOptions
  β :: Int = 1
  x_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  tband_func :: Function = (k, qxdim) -> qxdim # By default, have full density
  verbose :: Bool = false
end

# Make the Xin, Xsafe, and Xk
function makeXs!(model, inst :: QueryInstance, opts :: SplitSdpOptions)
  @assert inst isa SafetyInstance || (inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet)

  # Calculate variable dimensions
  ξvardims = makeξvardims(opts.β, inst, opts.tband_func)
  ξindim, ξsafedim, ξkdims = ξvardims
  Xvars = Dict()

  # Xin
  ξin = @variable(model, [1:ξindim])
  @constraint(model, ξin .>= 0)
  Xin = makeXin(ξin, inst.input, inst.ffnet)
  Xvars[:ξin] = ξin

  # Xsafe
  if inst isa SafetyInstance
    Xsafe = makeXsafe(inst.safety.S, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    ξsafe = @variable(model)
    @constraint(model, ξsafe >= 0)
    S = makeShyperplane(inst.reach_set.normal, ξsafe, inst.ffnet)
    Xsafe = makeXsafe(S, inst.ffnet)
    Xvars[:ξsafe] = ξsafe
  else
    error("unrecognized instance: " * string(inst))
  end

  # The Xks
  Xs = Vector{Any}()
  num_cliques = inst.ffnet.K - opts.β - 1
  for k = 1:(num_cliques+1)
    ξk = @variable(model, [1:ξkdims[k]])
    @constraint(model, ξk .>= 0)
    qxdim = Qxdim(k, opts.β, inst.ffnet.zdims)
    xqinfo = Xqinfo(
      ffnet = inst.ffnet,
      ϕout_intv = selectϕoutIntervals(k, opts.β, opts.x_intvs),
      slope_intv = selectSlopeIntervals(k, opts.β, opts.slope_intvs),
      tband = opts.tband_func(k, qxdim))
    Xk = makeXqξ(k, opts.β, ξk, xqinfo)
    Xvars[Symbol("ξ" * string(k))] = ξk
    push!(Xs, Xk)
  end

  return (Xin, Xsafe, Xs), Xvars, ξvardims
end

# Since the safety and verification problem many similarities, use a generic set up function 
function setup!(model, inst :: QueryInstance, opts :: SplitSdpOptions)
  @assert inst isa SafetyInstance || (inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet)
  setup_start_time = time()

  # Construct the Xs and set up Zβ
  zdims = inst.ffnet.zdims
  E1 = E(1, zdims)
  EK = E(inst.ffnet.K, zdims)
  Ea = E(inst.ffnet.K+1, zdims)
  Ein = [E1; Ea]
  Esafe = [E1; EK; Ea]

  (Xin, Xsafe, Xs), Xvars, ξvardims = makeXs!(model, inst, opts)
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert length(Xs) == num_cliques + 1

  Zβ = (Ein' * Xin * Ein) + (Esafe' * Xsafe * Esafe)
  for k in 1:(num_cliques+1)
    Ekβ = E(k, opts.β, zdims)
    EXk = [Ekβ; Ea]
    Zβ = Zβ + (EXk' * Xs[k] * EXk)
  end

  # Set up the Zks and do the NSD constraints
  Zs = Vector{Any}()
  for k = 1:num_cliques
    Ckdim = size(Ec(k, opts.β, zdims))[1]
    Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    @SDconstraint(model, Zk <= 0)
    push!(Zs, Zk)
  end

  # Assert the equality
  Zksum = sum(Ec(k, opts.β, zdims)' * Zs[k] * Ec(k, opts.β, zdims) for k in 1:num_cliques)
  @constraint(model, Zβ .== Zksum)

  # If we have a hyperplane reachability instance, additionally have an objective
  if inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    @objective(model, Min, Xvars[:ξsafe])
  end

  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, Xvars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts :: SplitSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = round(summary.solve_time, digits=3)
  if opts.verbose; println("solve time: " * string(solve_time)) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, solve_time
end

# The interface to call
function run(inst :: QueryInstance, opts :: SplitSdpOptions)
  # Appropriate number of cliques
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert num_cliques >= 1

  # Our tband_func does something reasonable
  qxdims = [Qxdim(k, opts.β, inst.ffnet.zdims) for k in 1:(num_cliques+1)]
  @assert all(k -> opts.tband_func(k, qxdims[k]) >= 0, 1:(num_cliques+1))

  # Start stuff
  total_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_REL_GAP" => 1e-6,
    "INTPNT_CO_TOL_PFEAS" => 1e-6,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  # Call the setup
  _, vars, setup_time = setup!(model, inst, opts)

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


# For debugging purposes we export more than what is needed
export SplitSdpOptions
export setup, solve!, run

end # End module

