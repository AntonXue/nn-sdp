using LinearAlgebra
using Parameters
using JuMP
using MosekTools
using Dualization
using Printf
using Dates

using ..MyLinearAlgebra
using ..MyNeuralNetwork
using ..Qc

# Default Mosek options
DEEPSDP_DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true)

# Options
@with_kw struct DeepSdpOptions <: QueryOptions
  max_solve_time::Float64 = 60.0 * 20
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = false
  verbose::Bool = false
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::DeepSdpOptions)
  setup_start_time = time()

  # Make the components
  MinP, Pvars = makeMinP!(model, query.input, query.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, query.qcinfos, query.ffnet, opts)
  MoutS = makeMoutS!(model, query.safety.S, query.ffnet, opts)

  # Now the LMI
  Z = MinP + MmidQ + MoutS
  @constraint(model, -Z in PSDCone())

  # Compute statistics and return
  vars = merge(Pvars, Qvars)
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Hyperplane reachability setup
function setupHplaneReach!(model, query::ReachQuery, opts::DeepSdpOptions)
  setup_start_time = time()
  @assert query.reach isa HplaneReachSet

  # Make the MinP and MmidQ first
  MinP, Pvars = makeMinP!(model, query.input, query.ffnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, query.qcinfos, query.ffnet, opts)

  # now setup MoutS
  @variable(model, h)
  Svars = Dict(:γh => h)
  S = makeShplane(query.reach.normal, h, query.ffnet)
  MoutS = makeMoutS!(model, S, query.ffnet, opts)

  # Set up the LMI and objective
  Z = MinP + MmidQ + MoutS
  @constraint(model, -Z in PSDCone())
  @objective(model, Min, h)

  # Calculate stuff and return
  vars = merge(Pvars, Qvars, Svars)
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts::DeepSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, summary.solve_time
end

# The interface to call
function runQuery(query::Query, opts::DeepSdpOptions)
  total_start_time = time()

  # Set up the model and add solver options, with the defaults first
  model = opts.use_dual ? Model(dual_optimizer(Mosek.Optimizer)) : Model(Mosek.Optimizer)
  pre_mosek_opts = opts.include_default_mosek_opts ? DEEPSDP_DEFAULT_MOSEK_OPTS : Dict()
  todo_mosek_opts = merge(pre_mosek_opts, opts.mosek_opts)
  for (k, v) in todo_mosek_opts; set_optimizer_attribute(model, k, v) end

  # Delegate the appropriate call depending on the kind of querylem
  if query isa SafetyQuery
    _, vars, setup_time = setupSafety!(model, query, opts)
  elseif query isa ReachQuery && query.reach isa HplaneReachSet
    _, vars, setup_time = setupHplaneReach!(model, query, opts)
  else
    error("\tunrecognized query: $(query)")
  end

  # Get ready to return
  if opts.verbose; println("\tabout to solve; now: $(now())") end

  summary, values, solve_time = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose;
    @printf("\tsetup: %.3f \tsolve: %.3f \ttotal: %.3f \tvalue: %.4e (%s)\n",
            setup_time, solve_time, total_time,
            objective_value(model), summary.termination_status)
  end
  return QuerySolution(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = solve_time)
end

