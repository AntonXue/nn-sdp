# Implement ADMM
module AdmmDeepSdp

using ..Header
using ..Common
using ..SplitDeepSdpB: Zk # Helpful to re-use
using LinearAlgebra


# Initialize zero parameters of the appropriate size
function initParams(inst :: VerificationInstance)
  @assert inst.net isa FeedForwardNetwork
  ffnet = inst.net
  input = inst.input
  K = ffnet.K
  xdims = ffnet.xdims
  zdims = [xdims[1:K]; 1]

  # The γ dimension
  γdims = Vector{Int}(zeros(K))
  for k = 1:K-1; γdims[k] = (xdims[k+1] * xdims[k+1]) + (2 * xdims[k+1]) end
  if input isa BoxConstraint
    γdims[K] = xdims[1]
  elseif input isa PolytopeConstraint
    γdims[K] = xdims[1] * xdims[1]
  else
    error("AdmmSdp:initParams: unsupported input " * string(input))
  end

  # All the variables that matter to each ωk
  γ = randn(sum(γdims))
  # γ = zeros(sum(γdims))

  # ωk are the 3-clique projections of γ, where ωk = ωs[k]
  ωs = [Hc(k, γdims) * γ for k in 1:K]

  # vk are the vectorized versions "Zk" matrices
  vs = Vector{Any}()
  for k = 1:K; push!(vs, zeros(length(zk(k, γdims, ωs[k], zdims, inst.input, inst.safety, ffnet)))) end

  # λk are the dual vars for vk = zk(ωk)
  λ = [zeros(length(v[k])) for k in 1:K]

  # μk are the dual vars for ωk = Hk * ω
  μ = [zeros(length(ωs[k])) for k in 1:K]

  return (γdims, γ, ωs, vs, λs, μs)
end

# The vectorized version
function zk(k, γdims, ωk, zdims, input, safety, ffnet)
  _Zk = Zk(k, γdims, ωk, zdims, input, safety, ffnet)
  return _Zk[:]
end

# Replacing the ωk with 1's should work to get the derivative
function ∂zk(k, γdims, ωk, zdims, input, safety, ffnet)
  _∂Zk = Zk(k, γdims, ones(length(ωk)), zdims, input, safety, ffnet)
  return _∂Zk[:]
end

# Project onto the non-negative orthant
function projectΩ(γ)
  return max.(abs.(γ), 0)
end

# The γ update
function stepγ(γdims, ωs, μs, ρ, dinv)
  K = length(γdims)
  tmp = [Hc(k, γdims)' * (ωs[k] + (μs[k] / ρ)) for k in 1:K]
  tmp = dinv .* sum(tmp)
  return projectΩ(tmp)
end

# Project a vector onto the negative semidefinite cone
function projectNsd(vk)
  dim = Int(round(sqrt(length(vk)))) # :)
  @assert length(vk) == dim * dim
  tmp = reshape(vk, (dim, dim))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  return tmp[:]
end

# The vk update
function stepvk(k, γdims, ωk, λk, zdims, ρ, input, safety, ffnet)
  tmp = zk(k, γdims, ωk, zdims, input, safety, ffnet)
  tmp = tmp - (λk / ρ)
  return projectNsd(tmp)
end

# The ωk update
function stepωk(k, γdims, ωk, zdims, ρ, input, safety, ffnet)
  tmp = ∂zk(k, γdims, ωk, zdims, input, safety, ffnet)
  tmp = Hc(k, γdims) * ω + v[k] - tmp
  tmp = tmp + ((λ[k] - μ[k])) # FIXME
end

# The λk update
function stepλk(k, γdims, ωk, zdims, ρ, input, safety, ffnet)
  tmp = zk(k, γdims, ωk, zdims, input, safety, ffnet)
  tmp = λ[k] + (tmp / ρ)
  return tmp
end

# The μk update
function stepμk(k, γdims)
  tmp = (ω[k] - Hc(k, γdims) * ω)
  tmp = μ[k] + (tmp / ρ)
  return tmp
end

#
function stepX()
end

#
function stepY()
end

#
function stepZ()
end

#
function stopcond(ω, vk, ωk, λk, μk)
  return False
end


#=
function admm



  this_ω
  this_vk
  this_ωk
  this_λk
  this_μk
  for t = 1:T
    next_ω, next_vk = stepX
    next_ωk = stepY
    next_λk, next_μk = stepZ

  end
end
=#


function run()
end

export stepω

end # End module
