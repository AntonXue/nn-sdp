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

# Options
@with_kw struct SplitSdpOptions
  β :: Int = 1
  x_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  slope_intervals :: Union{Nothing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}} = nothing
  verbose :: Bool = false
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
  Pvars = Dict(:γ => γ)
  return Xinit, Pvars
end

# Treat this as though it modifies the model
function makeSplitXsafe!(model, S, ffnet :: FeedForwardNetwork, opts :: SplitSdpOptions)
  Xsafe = makeXsafe(S, ffnet)
  return Xsafe
end

# Treat this as though it modifies the model
function makeSplitXk!(model, k :: Int, ffnet :: FeedForwardNetwork, opts :: SplitSdpOptions)
  qxdim = sum(ffnet.zdims[k+1:k+opts.β])
  if ffnet.type isa ReluNetwork
    λ_slope = @variable(model, [1:qxdim])
    τ_slope = @variable(model, [1:qxdim, 1:qxdim], Symmetric)
    η_slope = @variable(model, [1:qxdim])
    ν_slope = @variable(model, [1:qxdim])
    d_out = @variable(model, [1:qxdim])

    @constraint(model, λ_slope[1:qxdim] .>= 0)
    @constraint(model, τ_slope[1:qxdim, 1:qxdim] .>= 0)
    @constraint(model, η_slope[1:qxdim] .>= 0)
    @constraint(model, ν_slope[1:qxdim] .>= 0)
    @constraint(model, d_out[1:qxdim] .>= 0)

    vars = (λ_slope, τ_slope, η_slope, ν_slope, d_out)
    ϕout_intv = (opts.x_intervals isa Nothing) ? nothing : selectϕoutIntervals(k, opts.β, opts.x_intervals)
    slope_intv = (opts.slope_intervals isa Nothing) ? nothing : selectSlopeIntervals(k, opts.β, opts.slope_intervals)
    Xk = makeXk(k, opts.β, vars, ffnet, ϕout_intv=ϕout_intv, slope_intv=slope_intv)

    symk(v :: String) = Symbol(v * string(k))
    Qvars = Dict(symk("λ_slope") => λ_slope,
                  symk("τ_slope") => τ_slope,
                  symk("η_slope") => η_slope,
                  symk("ν_slope") => ν_slope,
                  symk("d_out") => d_out)
    return Xk, Qvars
  else
    error("unsupported network: " * string(ffnet))
  end
end

# Solve the safety problem by decomposing it into K-β-1 smaller LMIs
function setupSafety!(model, inst :: SafetyInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

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
  num_Xks = inst.ffnet.K - opts.β
  for k in 1:num_Xks
    if opts.verbose; println("making X[" * string(k) * "/" * string(num_Xks) * "]") end
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    Xk, Qkvars = makeSplitXk!(model, k, inst.ffnet, opts)
    Z = Z + (EXk' * Xk * EXk)
    merge!(Qvars, Qkvars)
  end

  # Select and assert that each Zk is NSD
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)
  Zscaled = Z .* Ωinv
  num_cliques = inst.ffnet.K - opts.β - 1
  for k = 1:num_cliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * Zscaled * Eck'
    @SDconstraint(model, Zk <= 0)
  end

  # Time it took to set up the problem
  vars = merge(Pvars, Qvars)
  setup_time = round(time() - setup_start_time, digits=2)
  if opts.verbose; println("setup time: " * string(setup_time)) end
  return model, vars, setup_time
end

# Solve the hyperplane reachability problem by decomposing it into K-β-1 smaller LMIs
function setupHyperplaneReachability!(model, inst :: ReachabilityInstance, opts :: SplitSdpOptions)
  setup_start_time = time()

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
  num_Xks = inst.ffnet.K - opts.β
  for k in 1:num_Xks
    if opts.verbose; println("making X[" * string(k) * "/" * string(num_Xks) * "]") end
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    Xk, Qkvars = makeSplitXk!(model, k, inst.ffnet, opts)
    Z = Z + (EXk' * Xk * EXk)
    merge!(Qvars, Qkvars)
  end

  # Select and assert that each Zk is NSD
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)
  Zscaled = Z .* Ωinv
  num_cliques = inst.ffnet.K - opts.β - 1
  for k = 1:num_cliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * Zscaled * Eck'
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
    "INTPNT_CO_TOL_DFEAS" => 1e-9))

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

