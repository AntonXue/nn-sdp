# A different implementation of the chordal decomposition of DeepSDP
# This one is based on projections partitions of a single large γ variable
module SplitDeepSDPb

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

#=
  Let Ya = Y[k-1] and Yb = Y[k+1]; we use the decomposition as in DeepSDPa
  Zk = [Ya[2,2] Yk[1,2] Yk[1,3]
                Yb[1,1] Yk[2,3]
                        Yk[3,3]]
=#

# ωk = [γ[k-1]; γ[k]; γ[k+1]]

function Zk11(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  a = (k == 1) ? ffnet.K : k - 1
  γa = G(k, 1, γd) * ωk
  _Ya = (a == ffnet.K) ? YγK(γa, input, safety, ffnet) : Yγk(a, γa, ffnet)
  return F(a, 2, zd) * _Ya * F(a, 2, zd)' # Ya[2,2]
end

function Zk12(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γd) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 1, zd) * _Yk * F(k, 2, zd)' # Yk[1,2]
end

function Zk13(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γd) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 1, zd) * _Yk * F(k, 3, zd)' # Yk[1,3]
end

function Zk22(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  b = (k == ffnet.K) ? 1 : k + 1
  γb = G(k, 3, γd) * ωk
  _Yb = (b == ffnet.K) ? YγK(γb, input, safety, ffnet) : Yγk(b, γb, ffnet)
  return F(b, 1, zd) * _Yb * F(b, 1, zd)' # Yb[1,1]
end

function Zk23(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γd) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 2, zd) * _Yk * F(k, 3, zd)' # Yk[2,3]
end

function Zk33(k, γd, ωk, zd, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γd) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 3, zd) * _Yk * F(k, 3, zd)' # Yk[3,3]
end

function Zk(k, γd, ωk, zd, input, safety, ffnet)
  _Zk11 = Zk11(k, γd, ωk, zd, input, safety, ffnet)
  _Zk12 = Zk12(k, γd, ωk, zd, input, safety, ffnet)
  _Zk13 = Zk13(k, γd, ωk, zd, input, safety, ffnet)
  _Zk22 = Zk22(k, γd, ωk, zd, input, safety, ffnet)
  _Zk23 = Zk23(k, γd, ωk, zd, input, safety, ffnet)
  _Zk33 = Zk33(k, γd, ωk, zd, input, safety, ffnet)
  Zk = [_Zk11 _Zk12 _Zk13; _Zk12' _Zk22 _Zk23; _Zk13' _Zk23' _Zk33]
  return Zk
end

#
function setup(ffnet, input, safety)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6
    ))

  xd = ffnet.xdims
  K = ffnet.K

  # First populate the γd dimensions
  γd = Vector{Int}(zeros(K))
  for k = 1:K-1; γd[k] = (xd[k+1] * xd[k+1]) + (2 * xd[k+1]) end

  if input isa BoxConstraint
    γd[K] = xd[1]
  elseif input isa PolytopeConstraint
    γd[K] = xd[1] * xd[1]
  else
    error("DeepSDPb:setup: unsupported input " * string(input))
  end

  # Now set up the varibales
  zd = [xd[1:K]; 1]
  @variable(model, ω[1:sum(γd)] >= 0)

  for k = 1:K
    ωk = Hc(k, γd) * ω
    _Zk = Zk(k, γd, ωk, zd, input, safety, ffnet)
    @SDconstraint(model, _Zk <= 0)
  end

  # When input isa PolytopeConstraint, also need diagonals == 0
  if input isa PolytopeConstraint
    γK = H(K, γd) * ω
    Γ = reshape(γ, (xd[1], xd[1]))
    for i = 1:xd[1]; @constraint(model, Γ[i,i] == 0) end
  end

  return model
end

# Run solver call
function solve(model)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(ffnet :: FeedForwardNetwork, input :: IC, safety :: SafetyConstraint) where {IC <: InputConstraint}
  start_time = time()
  model = setup(ffnet, input, safety)
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

#
export setup, solve, run

end # End module

