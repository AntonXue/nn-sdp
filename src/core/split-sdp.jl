# Implementation of the chordal decomposition of DeepSdp
module SplitSdp

using ..Header
using ..Common
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek
using Printf

# Options
@with_kw struct SplitSdpOptions
  β :: Int = 1
  x_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intvs :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  tband_func :: Function = (k, qxdim) -> qxdim # By default, have full density
  max_solve_time :: Float64 = 60.0
  verbose :: Bool = false
end

# Make the Xin, Xout, and Xk
function makeXs!(model, inst :: QueryInstance, opts :: SplitSdpOptions)
  @assert inst isa SafetyInstance || (inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet)

  # Calculate variable dimensions
  γvardims = makeγvardims(opts.β, inst, opts.tband_func)
  γindim, γoutdim, γkdims = γvardims
  Xvars = Dict()

  # Xin
  γin = @variable(model, [1:γindim])
  @constraint(model, γin .>= 0)
  Xin = makeXin(γin, inst.input, inst.ffnet)
  Xvars[:γin] = γin

  # Xout
  if inst isa SafetyInstance
    Xout = makeXout(inst.safety.S, inst.ffnet)
  elseif inst isa ReachabilityInstance && inst.reach_set isa HyperplaneSet
    γout = @variable(model)
    @constraint(model, γout >= 0)
    S = makeShyperplane(inst.reach_set.normal, γout, inst.ffnet)
    Xout = makeXout(S, inst.ffnet)
    Xvars[:γout] = γout
  else
    error(@sprintf("unrecognized instance: %s", inst))
  end

  # The Xks
  Xs = Vector{Any}()
  num_cliques = inst.ffnet.K - opts.β - 1
  for k = 1:(num_cliques+1)
    γk = @variable(model, [1:γkdims[k]])
    @constraint(model, γk .>= 0)
    qxdim = Qxdim(k, opts.β, inst.ffnet.zdims)
    xqinfo = Xqinfo(
      ffnet = inst.ffnet,
      ϕout_intv = selectϕoutIntervals(k, opts.β, opts.x_intvs),
      slope_intv = selectSlopeIntervals(k, opts.β, opts.slope_intvs),
      tband = opts.tband_func(k, qxdim))
    Xk = makeXqγ(k, opts.β, γk, xqinfo)
    Xvars[Symbol("γ" * string(k))] = γk
    push!(Xs, Xk)
  end

  return (Xin, Xout, Xs), Xvars, γvardims
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
  Eout = [E1; EK; Ea]

  (Xin, Xout, Xs), Xvars, γvardims = makeXs!(model, inst, opts)
  num_cliques = inst.ffnet.K - opts.β - 1
  @assert length(Xs) == num_cliques + 1

  Zβ = (Ein' * Xin * Ein) + (Eout' * Xout * Eout)
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
    @objective(model, Min, Xvars[:γout])
  end

  setup_time = time() - setup_start_time
  if opts.verbose; @printf("setup time: %.3f\n", setup_time) end
  return model, Xvars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts :: SplitSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = summary.solve_time
  if opts.verbose; @printf("solve time: %.3f\n", solve_time) end
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
    "MSK_DPAR_OPTIMIZER_MAX_TIME" => opts.max_solve_time,
    "INTPNT_CO_TOL_REL_GAP" => 1e-6,
    "INTPNT_CO_TOL_PFEAS" => 1e-6,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  # Call the setup
  _, vars, setup_time = setup!(model, inst, opts)

  # Get ready to return
  summary, values, solve_time = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose; @printf("total time: %.3f\n", total_time) end
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

