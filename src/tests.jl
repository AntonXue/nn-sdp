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


# Test the Z similarity
function testZscaling(inst :: QueryInstance, opts :: SplitSdpOptions, verbose = true)
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

  num_Xks = inst.ffnet.K - opts.β
  for k in 1:num_Xks
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
  num_cliques = inst.ffnet.K - opts.β - 1
  for k = 1:num_cliques
    Eck = Ec(k, opts.β, inst.ffnet.zdims)
    Zk = Eck * Zscaled * Eck'
    Zrecons = Zrecons + Eck' * Zk * Eck'
  end

  maxdiff = maximum(abs.(Z - Zrecons))
  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13
end

# Test the Yk decomposition
function testY(inst :: SafetyInstance, opts :: SplitSdpOptions, verbose = true)
  ffnet = inst.ffnet
  input = inst.input
  safety = inst.safety

  if input isa BoxInput
    ξinit = abs.(randn(ffnet.xdims[1]))
  elseif input isa PolytopeInput
    ξinit = abs.(randn(ffnet.xdims[1]^2))
  else
    error("unsupported input: " * string(input))
  end

  # Set up all the ξ1, ..., ξp, ξq variables first
  ξs = Vector{Any}()
  num_Xks = inst.ffnet.K - opts.β
  for k = 1:num_Xks
    # The size of each qxdim
    qxdim = sum(ffnet.zdims[k+1:k+opts.β])

    # Variables include:
    #   λ_slope: qxdim
    #   τ_slope: qxdim x qxdim
    #   η_slope: qxdim
    #   ν_slope: qxdim
    #   d_out: qxdim
    ξkdim = (qxdim)^2 + (4 * qxdim)
    ξk = abs.(randn(ξkdim))
    push!(ξs, ξk)
  end

  # Some helpful block index matrices
  E1 = E(1, ffnet.zdims)
  EK = E(ffnet.K, ffnet.zdims)
  Ea = E(ffnet.K+1, ffnet.zdims)

  # Construct the Z via X first
  Xinit = makeXinitξ(ξinit, input, ffnet)
  Xsafe = makeXsafeξ(safety, ffnet)
  Einit = [E1; Ea]
  Esafe = [E1; EK; Ea]
  ZXs = (Einit' * Xinit * Einit) + (Esafe' * Xsafe * Esafe)
  for k = 1:num_Xks
    Xk = makeXkξ(k, opts.β, ξs[k], ffnet)
    Ekβ = E(k, opts.β, ffnet.zdims)
    EXk = [Ekβ; Ea]
    ZXs = ZXs + (EXk' * Xk * EXk)
  end

  # Now construct Z via the Ys
  num_cliques = inst.ffnet.K - opts.β - 1
  if num_cliques == 1
    γ1 = (ξinit, ξs[1], ξs[2])
    Y1 = makeSafetyYk(1, opts.β, γ1, inst)
    Ec1 = Ec(1, opts.β, ffnet.zdims)
    ZYs = Ec1' * Y1 * Ec1

  else
    Ys = Vector{Any}()
    for k = 1:num_cliques
      if k == 1
        γ1 = (ξinit, ξs[1])
        Y1 = makeSafetyYk(1, opts.β, γ1, inst)
        push!(Ys, Y1)
      elseif k == num_cliques
        γp = (ξs[k], ξs[k+1])
        Yp = makeSafetyYk(k, opts.β, γp, inst)
        push!(Ys, Yp)
      else
        Yk = makeSafetyYk(k, opts.β, ξs[k], inst)
        push!(Ys, Yk)
      end
    end

    ZYs = sum(Ec(k, opts.β, ffnet.zdims)' * Ys[k] * Ec(k, opts.β, ffnet.zdims) for k in 1:num_cliques)
  end

  maxdiff = maximum(abs.(ZXs - ZYs))
  if verbose; println("maxdiff: " * string(maxdiff)) end
  @assert maxdiff <= 1e-13

end


end # End module

