module Tests

using ..Header
using ..Common
using ..DeepSdp
using ..SplitSdp

using LinearAlgebra
using Random
using JuMP
using Mosek
using MosekTools


function testXSumWhole(inst :: QueryInstance, verbose :: Bool = true)
  ffnet = inst.ffnet
  input = inst.input
  safety = inst.safety

  xdims = ffnet.xdims
  zdims = ffnet.zdims

  if input isa BoxInput
    γ = abs.(randn(xdims[1]))
  elseif input isa PolytopeInput
    γ = abs.(randn(xdims[1]^2))
  else
    error("unsupported input: " * string(input))
  end

  E1 = E(1, zdims)
  EK = E(ffnet.K, zdims)
  Ea = E(ffnet.K+1, zdims)

  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]

  # Set up the original formulation
  Xinit = makeXinit(γ, input, ffnet)
  MinP = Einit' * Xinit * Einit

  Xsafe = makeXsafe(safety, ffnet)
  MoutS = Esafe' * Xsafe * Esafe

  Tdim = sum(zdims[2:end-1])
  λ = abs.(randn(Tdim))
  τ = abs.(randn(Tdim, Tdim))
  η = abs.(randn(Tdim))
  ν = abs.(randn(Tdim))
  Q = makeQrelu(Tdim, λ, Λ, η, ν, inst.pattern)

  A = makeA(ffnet)
  b = makeb(ffnet)
  B = makeB(ffnet)
  _R11 = A
  _R12 = b
  _R21 = B
  _R22 = zeros(size(b))
  _R31 = zeros(1, sum(zdims[1:end-1]))
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  MmidQ = R' * Q * R

  origZ = MinP + MoutS + MmidQ

  # Another formulation
  Ekba = [E(1, ffnet.K-1, zdims); Ea]
  Xk = makeXk(1, ffnet.K-1, Λ, η, ν, ffnet, inst.pattern)

  Z = (Einit' * Xinit * Einit) + (Esafe' * Xsafe * Esafe) + (Ekba' * Xk * Ekba)

  maxdiff = maximum(abs.(origZ - Z))

  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13

end

# Test the Z similarity
function testZscaling(inst :: QueryInstance, opts :: SplitSdpOptions, verbose=true)
  ffnet = inst.ffnet
  input = inst.input
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)
  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]

  if inst.input isa BoxInput
    γ = abs.(randn(ffnet.xdims[1]))
  elseif inst.input isa PolytopeInput
    γ = abs.(randn(ffnet.xdims[1]^2))
  else
    error("unsupported input: " * string(input))
  end

  Xinit = makeXinit(γ, input, ffnet)

  S = Symmetric(abs.(randn(ffnet.xdims[1] + ffnet.xdims[end] + 1, ffnet.xdims[1] + ffnet.xdims[end] + 1)))
  Xsafe = makeXsafe(S, ffnet)
  
  Z = (Einit' * Xinit * Einit) * (Esafe' * Xsafe * Esafe)

  numXks = inst.ffnet.K - opts.β
  for k in 1:numXks
    qxdim = sum(ffnet.zdims[k+1:k+opts.β])
    Ekβ = E(k, opts.β, inst.ffnet.zdims)
    EXk = [Ekβ; Ea]
    λ = abs.(randn(qxdim))
    τ = Symmetric(abs.(randn(qxdim, qxdim)))
    η = abs.(randn(qxdim))
    ν = abs.(randn(qxdim))
    d = abs.(randn(qxdim))
    vars = (λ, τ, η, ν, d)
    Xk, _ = makeXk(k, opts.β, vars, ffnet)
    Z = Z + (EXk' * Xk * EXk)
  end

  # Select and assert that each Zk is NSD
  Ωinv = makeΩinv(opts.β, inst.ffnet.zdims)
  Zscaled = Z .* Ωinv
  Zrecons = zeros(size(Z))
  numCliques = inst.ffnet.K - opts.β - 1
  for k = 1:numCliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * Zscaled * Eck'
    Zrecons = Zrecons + Eck' * Zk * Eck'
  end

  maxdiff = maximum(abs.(Z - Zrecons))
  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
end

end # End module

