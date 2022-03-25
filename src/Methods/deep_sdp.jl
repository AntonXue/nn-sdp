using LinearAlgebra
using Parameters
using JuMP
using MosekTools
using Dualization
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork
using ..Qc

# Default Mosek options
DEEPSDP_DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true)

# Options
@with_kw struct DeepSdpOptions
  max_solve_time::Float64 = 60.0
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = false
  verbose::Bool = false
end

# Computes the MinP matrix. Treat this function as though it modifies the model
function makeMinP!(model, input::InputConstraint, nnet::NeuralNetwork, opts::DeepSdpOptions)
  if input isa BoxInput
    @variable(model, γin[1:nnet.zdims[1]] >= 0)
  elseif input isa PolytopeInput
    @variable(model, γin[1:nnet.zdims[1]^2] >= 0)
  else
    error("unsupported input constraints: $(input)")
  end
  E1 = E(1, nnet.zdims)
  Ea = E(nnet.K+1, nnet.zdims)
  Ein = [E1; Ea]
  Pin = makePin(γin, input, nnet)
  MinP = Ein' * Pin * Ein
  Pvars = Dict(:γin => γin)
  return MinP, Pvars
end

# Computes the MoutS matrix. Treat this function as though it modifies the model
function makeMoutS!(model, S, nnet::NeuralNetwork, opts::DeepSdpOptions)
  E1 = E(1, nnet.zdims)
  EK = E(nnet.K, nnet.zdims)
  Ea = E(nnet.K+1, nnet.zdims)
  Eout = [E1; EK; Ea]
  Sout = makeSout(S, nnet)
  MoutS = Eout' * Sout * Eout
  return MoutS
end

# Make the MmidQ matrix. Treat this function as though it modifies the model
function makeMmidQ!(model, qcinfos::Vector{QcInfo}, nnet::NeuralNetwork, opts::DeepSdpOptions)
  Qvars = Dict()
  Qs = Vector{Any}()
  for (i, qcinfo) in enumerate(qcinfos)
    # TODO: check whether a particular QC can be used
    γidim = vardim(qcinfo)
    γi = @variable(model, [1:γidim])
    Qvars[Symbol(:γ, i)] = γi
    @constraint(model, γi .>= 0)
    Q = makeQc(γi, qcinfo)
    push!(Qs, Q)
  end
  Q = sum(Qs)

  _R11 = makeA(nnet)
  _R12 = makeb(nnet)
  _R21 = makeB(nnet)
  _R22 = zeros(size(_R21)[1])
  _R31 = zeros(1, size(_R21)[2])
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  MmidQ = R' * Q * R
  return MmidQ, Qvars
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::DeepSdpOptions)
  setup_start_time = time()

  # Make the components
  MinP, Pvars = makeMinP!(model, query.input, query.nnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, query.qcinfos, query.nnet, opts)
  MoutS = makeMoutS!(model, query.output.S, query.nnet, opts)

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
  MinP, Pvars = makeMinP!(model, query.input, query.nnet, opts)
  MmidQ, Qvars = makeMmidQ!(model, query.qcinfos, query.nnet, opts)

  # now setup MoutS
  @variable(model, h)
  Svars = Dict(:h => h)
  S = makeShplane(query.reach.normal, h, query.nnet)
  MoutS = makeMoutS!(model, S, query.nnet, opts)

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
    error("unrecognized query: $(query)")
  end

  # Get ready to return
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

