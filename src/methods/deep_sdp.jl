using LinearAlgebra
using Parameters
using JuMP
using MosekTools
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork
using ..Qc

# Options
@with_kw struct DeepSdpOptions
  max_solve_time :: Float64 = 60.0
  verbose :: Bool = false
end

# Computes the MinP matrix. Treat this function as though it modifies the model
function makeMinP!(model, input :: InputConstraint, nnet :: NeuralNetwork, opts :: DeepSdpOptions)
  if input isa BoxInput
    @variable(model, γin[1:nnet.zdims[1]] >= 0)
  elseif input isa PolytopeInput
    @variable(model, γin[1:nnet.zdims[1]^2] >= 0)
  else
    error(@sprintf("unsupported input constraints: %s", input))
  end
  E1 = E(1, nnet.zdims)
  Ea = E(nnet.K+1, nnet.zdims)
  Ein = [E1; Ea]
  Xin = makeXin(γin, input, nnet)
  MinP = Ein' * Xin * Ein
  Pvars = Dict(:γin => γin)
  return MinP, Pvars
end

# Computes the MoutS matrix. Treat this function as though it modifies the model
function makeMoutS!(model, S, nnet :: NeuralNetwork, opts :: DeepSdpOptions)
  E1 = E(1, nnet.zdims)
  EK = E(nnet.K, nnet.zdims)
  Ea = E(nnet.K+1, nnet.zdims)
  Eout = [E1; EK; Ea]
  Xout = makeXout(S, nnet)
  MoutS = Eout' * Xout * Eout
  return MoutS
end

# Make the MmidQ matrix. Treat this function as though it modifies the model
function makeMmidQ!(model, qcinfos :: Vector{QcInfo}, nnet :: NeuralNetwork, opts :: DeepSdpOptions)
  Qvars = Dict()
  Qs = Vector{Any}()
  for (i, qcinfo) in enumerate(qcinfos)
    # TODO: check whether a particular QC can be used
    γidim = vardim(qcinfo)
    γi = @variable(model, [1:γidim])
    Qvars[Symbol(:γ, i)] = γi
    @constraint(model, γi >= 0)
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
function setupSafety!(model, prob :: SafetyProblem, opts :: DeepSdpOptions)
  setup_start_time = time()

  # Make the components
  MinP, Pvars = makeMinP!(model, prob.input, prob.nnet, opts)
  MmidQ, Qvars = makeMidQ!(model, prob.qcinfos, prob.nnet, opts)
  MoutS = makeMoutS!(model, prob.safety.S, prob.nnet, opts)

  # Now the LMI
  Z = MinP + MmidQ + MoutS
  @SDconstraint(model, Z <= 0)

  # Compute statistics and return
  vars = merge(Pvars, Qvars)
  setup_time = time() - setup_star_time
  return model, vars, setup_time
end


# Solve a model that is ready
function solve!(model, vars, opts :: DeepSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  solve_time = summary.solve_time
  if opts.verbose; @printf("\t solve time: %.3f\n", solve_time) end
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, solve_time
end

# The interface to call
function run(prob :: Problem, opts :: DeepSdpOptions)
  total_start_time = time()
  model = Model(Mosek.Optimizer)
  set_optimizer_attribute(model, "QUIET", true)
  set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", opts.max_solve_time)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_REL_GAP", 1e-6)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_PFEAS", 1e-6)
  set_optimizer_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-6)

  # Delegate the appropriate call depending on the kind of problem
  if prob isa SafetyProblem
    _, vars, setup_time = setupSafety!(model, prob, opts)
  else
    error(@sprintf("unrecognized problem: %s", prob))
  end

  # Get ready to return
  summary, values, solve_time = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose; @printf("\ttotal time: %.3f\n", total_time) end
  return SolutionOutput(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = summary.termination_status,
    total_time = total_time,
    setup_time = setup_time,
    solve_time = solve_time)
end

