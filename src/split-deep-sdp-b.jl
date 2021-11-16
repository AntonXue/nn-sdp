# A different implementation of the chordal decomposition of DeepSdp
# This one is based on projections partitions of a single large γ variable
module SplitDeepSdpB

using ..Header
using ..Common
using LinearAlgebra
using JuMP
using MosekTools
using Mosek

#=
  Let a = k-1 and b = k+1; we use the decomposition as in DeepSdpa
  Zk = [Ya[2,2] Yk[1,2] Yk[1,3]
                Yb[1,1] Yk[2,3]
                        Yk[3,3]]
=#

# ωk = [γa; γk; γb], with block-wise indexing of course!

function Zk11(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  a = (k == 1) ? ffnet.K : k - 1
  γa = G(k, 1, γdims) * ωk
  _Ya = (a == ffnet.K) ? YγK(γa, input, safety, ffnet) : Yγk(a, γa, ffnet)
  return F(a, 2, zdims) * _Ya * F(a, 2, zdims)' # Ya[2,2]
end

function Zk12(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γdims) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 1, zdims) * _Yk * F(k, 2, zdims)' # Yk[1,2]
end

function Zk13(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γdims) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 1, zdims) * _Yk * F(k, 3, zdims)' # Yk[1,3]
end

function Zk22(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  b = (k == ffnet.K) ? 1 : k + 1
  γb = G(k, 3, γdims) * ωk
  _Yb = (b == ffnet.K) ? YγK(γb, input, safety, ffnet) : Yγk(b, γb, ffnet)
  return F(b, 1, zdims) * _Yb * F(b, 1, zdims)' # Yb[1,1]
end

function Zk23(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γdims) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 2, zdims) * _Yk * F(k, 3, zdims)' # Yk[2,3]
end

function Zk33(k, ωk, γdims, zdims, input, safety, ffnet :: FeedForwardNetwork)
  γk = G(k, 2, γdims) * ωk
  _Yk = (k == ffnet.K) ? YγK(γk, input, safety, ffnet) : Yγk(k, γk, ffnet)
  return F(k, 3, zdims) * _Yk * F(k, 3, zdims)' # Yk[3,3]
end

function Zk(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk11 = Zk11(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk12 = Zk12(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk13 = Zk13(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk22 = Zk22(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk23 = Zk23(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk33 = Zk33(k, ωk, γdims, zdims, input, safety, ffnet)
  Zk = [_Zk11 _Zk12 _Zk13; _Zk12' _Zk22 _Zk23; _Zk13' _Zk23' _Zk33]
  return Zk
end

# Set up the model
function setup(inst :: VerificationInstance)
  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  input = inst.input
  safety= inst.safety
  xdims = ffnet.xdims
  K = ffnet.K

  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-6
    ))

  # Calculate the dimensions of γ and store them in γdims
  if ffnet.nettype isa ReluNetwork
    γdims = Vector{Int}(zeros(K))
    for k = 1:K-1; γdims[k] = (xdims[k+1] * xdims[k+1]) + (2 * xdims[k+1]) end
  else
    error("DeepSdpB:setup: unsupported network " * string(ffnet))
  end

  # The last dimension is dependent on the input constraint
  if input isa BoxConstraint
    γdims[K] = xdims[1]
  elseif input isa PolytopeConstraint
    γdims[K] = xdims[1] * xdims[1]
  else
    error("DeepSdpB:setup: unsupported input " * string(input))
  end

  # zdims denotes the dimensions that partition the big Z matrix
  zdims = [xdims[1:K]; 1]
  @variable(model, γ[1:sum(γdims)] >= 0) # Really γ = ω notation-wise

  for k = 1:K
    ωk = Hc(k, γdims) * γ
    _Zk = Zk(k, ωk, γdims, zdims, input, safety, ffnet)
    @SDconstraint(model, _Zk <= 0)
  end

  # When input isa PolytopeConstraint, also need diagonals == 0 ...maybe not
  # if input isa PolytopeConstraint
  #   γK = H(K, γdims) * γ
  #   Γ = reshape(γ, (xdims[1], xdims[1]))
  #   for i = 1:xdims[1]; @constraint(model, Γ[i,i] == 0) end
  # end

  return model
end

# Run solver call
function solve(model)
  optimize!(model)
  return solution_summary(model)
end

# The interface to call
function run(inst :: VerificationInstance)
  start_time = time()
  model = setup(inst)
  summary = solve(model)
  total_time = time() - start_time

  output = SolutionOutput(
            model = model,
            summary = summary,
            status = string(summary.termination_status),
            total_time = total_time,
            solve_time = summary.solve_time)
  return output
end

#
export setup, solve, run, Zk

end # End module

