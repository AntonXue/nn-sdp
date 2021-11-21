# Implementation of the DeepSdp algorithm
module DeepSdp

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Set up the jump model
function setup(inst :: VerificationInstance, opts :: VerificationOptions)
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
    @variable(model, Γ[1:xdims[1], 1:xdims[1]] >= 0)
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
      Λ = @variable(model, [1:qxdim, 1:qxdim])
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
  return model
end

# Run the optimization scheme and query the solution summary
function solve!(model)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: VerificationInstance, opts :: VerificationOptions)
  start_time = time()
  model = setup(inst, opts)
  summary = solve!(model)
  total_time = time() - start_time

  output = SolutionOutput(
            model = model,
            summary = summary,
            status = string(summary.termination_status),
            total_time = total_time,
            solve_time = summary.solve_time)
  return output
end

# For debugging purposes we export more than what is needed
export setup, solve!, run

end # End module

