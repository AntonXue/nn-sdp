# Implementation of LipSdp
module LipSdp

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

function setup(ffnet :: FeedForwardNetwork)
  setup_start_time = time()

  xdims = ffnet.xdims
  K = ffnet.K

  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6))

  λdim = sum(xdims[2:K])

  Λ = @variable(model, [1:λdim, 1:λdim], Symmetric)
  ρ = @variable(model)

  model[:Λ] = Λ
  model[:ρ] = ρ

  @constraint(model, Λ[1:λdim, 1:λdim] .>= 0)
  @constraint(model, ρ >= 0)

  T = Tλ(λdim, Λ)
  A, _, B = Qsides(1, ffnet, stride=K-1)

  println("size T: " * string(size(T)))
  println("size A: " * string(size(A)))
  println("size B: " * string(size(B)))

  # Set up the M matrix, the first part
  α = 0.0
  β = 1.0
  
  _R11 = -2 * α * β * T
  _R12 = (α + β) * T
  _R22 = -2 * T
  R = [_R11 _R12; _R12' _R22]
  M1 = [A; B]' * R * [A; B]

  # The second part
  F = [E(1, xdims[1:K]); E(K, xdims[1:K])]
  WK = ffnet.Ms[K][1:end, 1:end-1]
  _S11 = -ρ * I(xdims[1])
  _S12 = zeros(xdims[1], xdims[K])
  _S22 = WK' * WK
  S = [_S11 _S12; _S12' _S22]
  M2 = F' * S * F

  # Now add them together and assert NSD
  M = M1 + M2

  @SDconstraint(model, M <= 0)
  return model
end

function solve!(model)
  println("solve! called")
  solve_start_time = time()
  optimize!(model)
  return solution_summary(model)
end

function run(ffnet :: FeedForwardNetwork)
  start_time = time()
  model = setup(ffnet)
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

export setup, solve!, run

end
