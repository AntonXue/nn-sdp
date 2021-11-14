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
  xd = ffnet.xdims
  zd = [xd[1:K]; 1]

  # The γ dimension
  γd = Vector{Int}(zeros(K))
  for k = 1:K-1; γd[k] = (xd[k+1] * xd[k+1]) + (2 * xd[k+1]) end
  if input isa BoxConstraint
    γd[K] = xd[1]
  elseif input isa PolytopeConstraint
    γd[K] = xd[1] * xd[1]
  else
    error("AdmmSdp:initParams: unsupported input " * string(input))
  end

  # All the variables that matter to each ωk
  γ = randn(sum(γd))
  # γ = zeros(sum(γd))

  # ωk are the 3-clique projections of γ
  ω = [Hc(k, γd) * γ for k in 1:K]

  # vk are the vectorized versions "Zk" matrices
  v = Vector{Any}()
  for k = 1:K; push!(v, zeros(length(zk(k, γd, ω[k], zd, inst.input, inst.safety, ffnet)))) end

  # λk are the dual vars for vk = zk(ωk)
  λ = [zeros(length(v[k])) for k in 1:K]

  # μk are the dual vars for ωk = Hk * ω
  μ = [zeros(length(ω[k])) for k in 1:K]

  return (γd, γ, ω, v, λ, μ)
end

# The vectorized version
function zk(k, γd, ωk, zd, input, safety, ffnet)
  _Zk = Zk(k, γd, ωk, zd, input, safety, ffnet)
  return _Zk[:]
end

# Replacing the ωk with 1's should work to get the derivative
function ∂zk(k, γd, ωk, zd, input, safety, ffnet)
  _∂Zk = Zk(k, γd, ones(length(ωk)), zd, input, safety, ffnet)
  return _∂Zk[:]
end

# Project onto the non-negative orthant
function projectΩ(ω)
  return max.(abs.(ω), 0)
end

# The ω update
function stepω(γd, ω, μ, ρ, dinv)
  K = length(γd)
  tmp = [Hc(k, γd)' * (ω[k] + (μ[k] / ρ)) for k in 1:K]
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
function stepvk(k, γd, ωk, λk, zd, ρ, input, safety, ffnet)
  tmp = zk(k, γd, ωk, zd, input, safety, ffnet)
  tmp = tmp - (λk / ρ)
  return projectNsd(tmp)
end

# The ωk update
function stepωk(k, γd, ωk, zd, ρ, input, safety, ffnet)
  tmp = ∂zk(k, γd, ωk, zd, input, safety, ffnet)
  tmp = Hc(k, γd) * ω + v[k] - tmp
  tmp = tmp + ((λ[k] - μ[k])) # FIXME
end

# The λk update
function stepλk(k, γd, ωk, zd, ρ, input, safety, ffnet)
  tmp = zk(k, γd, ωk, zd, input, safety, ffnet)
  tmp = λ[k] + (tmp / ρ)
  return tmp
end

# The μk update
function stepμk(k, γd)
  tmp = (ω[k] - Hc(k, γd) * ω)
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
