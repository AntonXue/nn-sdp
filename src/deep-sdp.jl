# Implementation of the DeepSdp algorithm
module DeepSdp

using ..Header
using ..Common
using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Construction method
abstract type SetupMethod end
struct SimpleSetup <: SetupMethod end

@with_kw struct DeepSdpOptions
  setupMethod :: SetupMethod
  verbose :: Bool = false
end

function setup_Simple(model, inst :: VerificationInstance, opts :: DeepSdpOptions)
  setup_start_time = time()

  ffnet = inst.net
  input = inst.input
  safety = inst.safety

  xdims = ffnet.xdims
  zdims = ffnet.zdims

  if input isa BoxConstraint
    @variable(model, γ[1:xdims[1]] >= 0)
    P = BoxP(input.xbot, input.xtop, γ)
  elseif input isa PolytopeConstraint
    @variable(model, Γ[1:xdims[1], 1:xdims[1]] >= 0, Symmetric)
    P = PolytopeP(input.H, input.h, Γ)
  else
    error("DeepSdp:setup: unsupported input " * string(input))
  end

  E1 = E(1, zdims)
  EK = E(ffnet.K, zdims)
  Ea = E(ffnet.K+1, zdims)

  Xinit = P
  Einit = [E1; Ea]

  Xsafe = makeXsafe(safety.S, ffnet)
  Esafe = [E1; EK; Ea]

  Z = Einit' * Xinit * Einit + Esafe' * Xsafe * Esafe

  for k = 1:(ffnet.K - inst.β)
    Tdim = sum(ffnet.zdims[(k+1) : (k+inst.β)])
    Λ = @variable(model, [1:Tdim, 1:Tdim], Symmetric)
    η = @variable(model, [1:Tdim])
    ν = @variable(model, [1:Tdim])

    @constraint(model, Λ[1:Tdim, 1:Tdim] .>= 0)
    @constraint(model, η[1:Tdim] .>= 0)
    @constraint(model, ν[1:Tdim] .>= 0)

    Xk = makeXk(k, inst.β, Λ, η, ν, ffnet, inst.pattern)
    Ekba = [E(k, inst.β, zdims); Ea]
    Z = Z + Ekba' * Xk * Ekba
  end

  @SDconstraint(model, Z <= 0)
  setup_time = time() - setup_start_time
  return model, setup_time
end

#=
# Set up the jump model
function setup(inst :: VerificationInstance, opts :: VerificationOptions)
  setup_start_time = time()

  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  input = inst.input
  safety = inst.safety
  xdims = ffnet.xdims
  zdims = ffnet.zdims
  K = ffnet.K
  stride = opts.stride
  p = K - stride
  @assert p >= 1

  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6
    ))

  if input isa BoxConstraint
    @variable(model, γ[1:xdims[1]] >= 0)
    P = BoxP(input.xbot, input.xtop, γ)
  elseif input isa PolytopeConstraint
    @variable(model, Γ[1:xdims[1], 1:xdims[1]] >= 0, Symmetric)
    P = PolytopeP(input.H, input.h, Γ)
  else
    error("DeepSdp:setup: unsupported input " * string(input))
  end

  # The input and safety matrices
  Yin = Yinput(P, ffnet, stride=stride)
  Ysafe = Ysafety(safety.S, ffnet, stride=stride)

  # The rest of the Ys
  Ys = Vector{Any}()
  if ffnet.nettype isa ReluNetwork
    for k in 1:p
      qxdim = sum(xdims[k+1:k+stride])
      Λ = @variable(model, [1:qxdim, 1:qxdim], Symmetric)
      η = @variable(model, [1:qxdim])
      ν = @variable(model, [1:qxdim])

      @constraint(model, Λ[1:qxdim, 1:qxdim] .>= 0)
      @constraint(model, η[1:qxdim] .>= 0)
      @constraint(model, ν[1:qxdim] .>= 0)

      Qk = Qrelu(qxdim, Λ, η, ν)
      Yk = Y(k, Qk, ffnet, stride=stride)
      push!(Ys, Yk)
    end
  else
    error("DeepSdp:setup: unsupported network " * string(ffnet))
  end

  @assert length(Ys) == p

  # Now setup big Z
  Ec1 = Ec(1, zdims, stride=stride)
  Ecp = Ec(p, zdims, stride=stride)
  Z = Ec1' * Yin * Ec1 + Ecp' * Ysafe * Ecp
  for k = 1:p
    Eck = Ec(k, zdims, stride=stride)
    Yk = Ys[k]
    println("Looping k[" * string(k) * "/" * string(p) * "], size(Yk): " * string(size(Yk)))
    Z += Eck' * Yk * Eck
  end

  @SDconstraint(model, Z <= 0)

  println("setup: returning with time: " * string(time() - setup_start_time))

  return model
end
=#

function setup(inst :: VerificationInstance, opts :: DeepSdpOptions)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))
  
  if opts.setupMethod isa SimpleSetup
    model, setup_time = setup_Simple(model, inst, opts)
  else
    error("unsupported setup method: " * string(opts.setupMethod))
  end

  println("setup time: " * string(setup_time))

  return model, setup_time
end

# Run the optimization scheme and query the solution summary
function solve!(model, opts :: DeepSdpOptions)
  solve_start_time = time()
  println("calling optimize")
  optimize!(model)
  solve_time = time() - solve_start_time
  println("optimize call done: " * string(solve_time))
  return solution_summary(model)
end

# The interface to call
function run(inst :: VerificationInstance, opts :: DeepSdpOptions)
  start_time = time()
  model, setup_time = setup(inst, opts)
  summary = solve!(model, opts)
  total_time = time() - start_time

  output = SolutionOutput(
            model = model,
            summary = summary,
            status = string(summary.termination_status),
            total_time = total_time,
            setup_time = setup_time,
            solve_time = summary.solve_time)
  return output
end

# For debugging purposes we export more than what is needed

export SetupMethod, SimpleSetup
export DeepSdpOptions
export setup, solve!, run

end # End module

