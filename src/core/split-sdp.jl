# Implementation of the chordal decomposition of DeepSdp
module SplitSdp

using ..Header
using ..Common
using ..Partitions
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
  tband_func :: Function = (k, qkxdim) -> qkxdim # By default, have full density
  verbose :: Bool = false
end

# Make the Ys
function makeYs!(model, inst :: QueryInstance, opts :: SplitSdpOptions)
  # Figure out the relevant γ dimensions
  γdims, ξvardims = makeγdims(opts.β, inst, opts.tband_func)
  Ys = Vector{Any}()
  Yvars = Dict()
  num_cliques = inst.ffnet.K - opts.β - 1

  yinfo = Yinfo(
    inst = inst,
    num_cliques = num_cliques,
    x_intvs = opts.x_intvs,
    slope_intvs = opts.slope_intvs,
    ξvardims = ξvardims,
    tband_func = opts.tband_func)

  # Make each Yk
  for k = 1:num_cliques
    γk = @variable(model, [1:γdims[k]])
    @constraint(model, γk[1:γdims[k]] .>= 0)
    Yk = makeYk(k, opts.β, γk, yinfo)
    push!(Ys, Yk)
    Yvars[Symbol("γ" * string(k))] = γk
  end
  return Ys, Yvars, ξvardims
end

# Do the safety instance
function setupSafety!(model, inst :: SafetyInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

  # Construct the Ys
  Ys, Yvars, ξvardims = makeYs!(model, inst, opts)
  Ωinvs = makeΩinvs(opts.β, inst.ffnet.zdims)
  num_cliques = inst.ffnet.K - opts.β - 1

  # Use these Yks to construct the Zks
  for k = 1:num_cliques
    Zk = makeZk(k, opts.β, Ys, Ωinvs, inst.ffnet.zdims)
    @SDconstraint(model, Zk <= 0)
  end

  # Ready to return
  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, Yvars, setup_time
end

# Do the safety instance
function setupSafetyNoSplit!(model, inst :: SafetyInstance, opts :: SplitSdpOptions)
  setup_start_time = time()
  println("\tsetupSafetyNoSplit!")

  # Construct the Ys
  Ys, Yvars, ξvardims = makeYs!(model, inst, opts)
  Ωinvs = makeΩinvs(opts.β, inst.ffnet.zdims)
  num_cliques = inst.ffnet.K - opts.β - 1

  @assert length(Ys) == num_cliques
  zdims = inst.ffnet.zdims
  Z = sum(Ec(k, opts.β, zdims)' * Ys[k] * Ec(k, opts.β, zdims) for k in 1:num_cliques)
  @SDconstraint(model, Z <= 0)

  # Ready to return
  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, Yvars, setup_time
end

#
function setupSafetyZVs!(model, inst :: SafetyInstance, opts :: SplitSdpOptions)
  setup_start_time = time()
  println("\tsetupSafetyZVs!")

  # Construct the Ys
  Ys, Yvars, ξvardims = makeYs!(model, inst, opts)
  num_cliques = inst.ffnet.K - opts.β - 1

  @assert length(Ys) == num_cliques
  zdims = inst.ffnet.zdims

  Zs = Vector{Any}()
  Vs = Vector{Any}()
  for k = 1:num_cliques
    Ckdim = size(Ec(k, opts.β, zdims))[1]
    Vk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
    Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)

    @constraint(model, Vk .== Zk)
    @SDconstraint(model, Zk <= 0)
    push!(Vs, Vk)
    push!(Zs, Zk)
  end

  Lhs = sum(Ec(k, opts.β, zdims)' * Vs[k] * Ec(k, opts.β, zdims) for k in 1:num_cliques)
  Rhs = sum(Ec(k, opts.β, zdims)' * Ys[k] * Ec(k, opts.β, zdims) for k in 1:num_cliques)
  @constraint(model, Lhs .== Rhs)

  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, Yvars, setup_time
end

# Do the hyperplane reachability instance
function setupHyperplaneReachability!(model, inst :: ReachabilityInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

  # Construct the Ys
  Ys, Yvars, ξvardims = makeYs!(model, inst, opts)
  Ωinvs = makeΩinvs(opts.β, inst.ffnet.zdims)
  num_cliques = inst.ffnet.K - opts.β - 1

  # Use these Yks to construct the Zks
  for k = 1:num_cliques
    Zk = makeZk(k, opts.β, Ys, Ωinvs, inst.ffnet.zdims)
    @SDconstraint(model, Zk <= 0)
  end

  ξsafe = spliceγ1(Yvars[:γ1], ξvardims)[2][1]
  vars = merge(Yvars, Dict(:h => ξsafe))
  @objective(model, Min, ξsafe)

  # Ready to return
  setup_time = round(time() - setup_start_time, digits=3)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
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

  # Delegate the appropriate call depending on our query instance
  if inst isa SafetyInstance
    # _, vars, setup_time = setupSafety!(model, inst, opts)
    # _, vars, setup_time = setupSafetyNoSplit!(model, inst, opts)
    _, vars, setup_time = setupSafetyZVs!(model, inst, opts)
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


# For debugging purposes we export more than what is needed
export SplitSdpOptions
export setup, solve!, run

end # End module

