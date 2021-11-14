# Implementation of the DeepSdp algorithm
module DeepSdp

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

# Set up the jump model
function setup(inst :: VerificationInstance)
  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  input = inst.input
  safety = inst.safety
  xd = ffnet.xdims
  K = ffnet.K

  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6
    ))

  if input isa BoxConstraint
    @variable(model, γ[1:xd[1]] >= 0)
    P = BoxP(input.xbot, input.xtop, γ)
  elseif input isa PolytopeConstraint
    @variable(model, Γ[1:xd[1], 1:xd[1]] >= 0)
    # for i in 1:xd[1]; @constraint(model, Γ[i,i] == 0) end
    P = PolytopeP(input.H, input.h, Γ)
  else
    error("DeepSdp:setup: unsupported input " * string(input))
  end

  Y = Vector{Any}()
  if ffnet.nettype isa ReluNetwork
    # Setup the variables
    for k = 1:K-1
      xdk1 = xd[k+1]
      Λ = @variable(model, [1:xdk1, 1:xdk1])
      ν = @variable(model, [1:xdk1])
      η = @variable(model, [1:xdk1])

      @constraint(model, Λ[1:xdk1, 1:xdk1] .>= 0)
      @constraint(model, ν[1:xdk1] .>= 0)
      @constraint(model, η[1:xdk1] .>= 0)

      Qk = Qrelu(Λ, ν, η)
      _Yk = Yk(k, Qk, ffnet)
      push!(Y, _Yk)
    end
    # The final YK
    _YK = YK(P, safety.S, ffnet)
    push!(Y, _YK)
  else
    error("DeepSdp:setup: unsupported network " * string(ffnet))
  end

  @assert length(Y) == ffnet.K

  # Now setup big Z
  zd = [xd[1:K]; 1]
  sumzd = sum(zd)
  Z = zeros(sumzd, sumzd)
  for k = 1:K
    Eck = Ec(k, zd)
    Z = Z + Eck' * Y[k] * Eck
  end

  @SDconstraint(model, Z <= 0)
  return model
end

# Run the optimization scheme and query the solution summary
function solve(model)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: VerificationInstance)
  start_time = time()
  model = setup(inst)
  summary = solve(model)
  end_time = time()
  total_time = end_time - start_time

  output = SolutionOutput(
            model = model,
            summary = summary,
            status = string(summary.termination_status),
            total_time = total_time,
            solve_time = summary.solve_time)
  return output
end

# For debugging purposes we export more than what is needed
export setup, solve, run

end # End module

