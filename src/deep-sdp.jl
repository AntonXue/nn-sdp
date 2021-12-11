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
  verbose :: Bool = false
end

# Computes the MinP matrix. Treat this function as though it modifies the model
function makeMinP!(model, input :: InputConstraint, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  if input isa BoxConstraint
    @variable(model, γ[1:ffnet.xdims[1]] >= 0)
  elseif input isa PolytopeConstraint
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

# Computes the MmidQ matrix. Treat this function as though it modifies the model
function makeMmidQ!(model, pattern :: TPattern, ffnet :: FeedForwardNetwork, opts :: DeepSdpOptions)
  Tdim = sum(ffnet.zdims[2:end-1])
  Λ = @variable(model, [1:Tdim, 1:Tdim], Symmetric)
  η = @variable(model, [1:Tdim])
  ν = @variable(model, [1:Tdim])

  @constraint(model, Λ[1:Tdim, 1:Tdim] .>= 0)
  @constraint(model, η[1:Tdim] .>= 0)
  @constraint(model, ν[1:Tdim] .>= 0)

  b = ffnet.K - 1
  MmidQ = makeXk(1, b, Λ, η, ν, ffnet, pattern)
  return MmidQ, Λ, η, ν
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

# Solve the safety verification problem instance
function solveSafety(inst :: SafetyInstance, opts :: DeepSdpOptions)
  total_start_time = time()
  setup_start_time = time()
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  # Make the different parts
  MinP, γ = makeMinP!(model, inst.input, inst.ffnet, opts)
  MmidQ, Λ, η, ν = makeMmidQ!(model, inst.pattern, inst.ffnet, opts)
  MoutS = makeMoutS!(model, inst.safety.S, inst.ffnet, opts)
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
  soln_dict = Dict(:γ => value.(γ), :Λ => value.(Λ), :η => value.(η), :ν => value.(ν))
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
function solveReachability(inst :: ReachabilityInstance, opts :: DeepSdpOptions)
  total_start_time = time()

  ds = Vector{Float64}()
  summaries = Vector{Any}()
  statuses = Vector{String}()
  soln_dicts = Vector{Any}()
  setup_times = Vector{Float64}()
  solve_times = Vector{Float64}()

  num_hplanes = length(inst.hplanes.normals)

  for (i, c) in enumerate(inst.hplanes.normals)
    iter_setup_start_time = time()

    model = Model(optimizer_with_attributes(
      Mosek.Optimizer,
      "QUIET" => true,
      "INTPNT_CO_TOL_DFEAS" => 1e-6))

    # Make the different parts
    MinP, γ = makeMinP!(model, inst.input, inst.ffnet, opts)
    MmidQ, Λ, η, ν = makeMmidQ!(model, inst.pattern, inst.ffnet, opts)

    @variable(model, d)

    hplaneS = makeHyperplaneS(c, d, inst.ffnet)
    MoutS = makeMoutS!(model, hplaneS, inst.ffnet, opts)
    Z = MinP + MmidQ + MoutS
    @SDconstraint(model, Z <= 0)
    @objective(model, Min, d)

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
    if opts.verbose; println("iter[" * string(i) * "/" * string(num_hplanes) * "] solve time: " * string(iter_solve_time)) end

    dopt = value(d)
    push!(ds, dopt)

    soln_dicti = Dict(:γ => value.(γ), :Λ => value.(Λ), :η => value.(η), :ν => value.(ν))
    push!(soln_dicts, soln_dicti)
  end

  # Prepare and return the output
  cds = [(c,d) for (c,d) in zip(inst.hplanes.normals, ds)]
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
  elseif inst isa ReachabilityInstance
    return solveReachability(inst, opts)
  else
    error("unrecognized query instance: " * string(inst))
  end
end

# For debugging purposes we export more than what is needed

# export SetupMethod, BigSetup
export DeepSdpOptions
export run

end # End module

