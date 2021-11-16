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
  # γ = randn(sum(γdims))
  γ = zeros(sum(γdims))

  # ωk are the 3-clique projections of γ, where ωk = ωs[k]
  ωs = [Hc(k, γdims) * γ for k in 1:K]

  # vk are the vectorized versions "Zk" matrices
  vs = Vector{Any}()
  for k = 1:K; push!(vs, zeros(length(zk(k, ωs[k], γdims, zdims, inst.input, inst.safety, ffnet)))) end

  # λk are the dual vars for vk = zk(ωk)
  λs = [zeros(length(vs[k])) for k in 1:K]

  # μk are the dual vars for ωk = Hk * γ
  μs = [zeros(length(ωs[k])) for k in 1:K]

  # A guess
  ρ = 1

  #
  params = (γ, vs, ωs, λs, μs, ρ, γdims)
  return params
end

# Computes some potentially expensive things
# function precompute(γ, ωs, γdims, zdims, input, safety, ffnet)
function precompute(params, inst)
  γ, vs, ωs, λs, μs, ρ, γdims = params

  input = inst.input
  safety = inst.safety
  ffnet = inst.net

  K = length(γdims)
  zdims = [ffnet.xdims[1:K]; 1]

  Ys = Vector{Any}()
  zs = Vector{Any}()
  Js = Vector{Any}()
  Jtzas = Vector{Any}()
  I_JtJ_invs = Vector{Any}()
  println("precompute: k loop start, total K = " * string(K))

  # Calculate the Ys
  for k = 1:K
    kstart_time = time()
    Yka = (k == K) ? YγK(zeros(γdims[k]), input, safety, ffnet) : Yγk(k, zeros(γdims[k]), ffnet)
    
    Ykparts = Vector{Any}()
    for j = 1:γdims[k]
      tmp = (k == K) ? YγK(e(j, γdims[k]), input, safety, ffnet) : Yγk(k, e(j, γdims[k]), ffnet)
      Ykj = tmp - Yka
      push!(Ykparts, Ykj)
    end
    push!(Ykparts, Yka)
    push!(Ys, Ykparts)

    k_time = time() - kstart_time
    println("precompute: k = " * string(k) * "/" * string(K) * ", time = " * string(k_time))
  end

  # Populate the Zk
  for k = 1:K
    #=
    a = (k == 1) ? K : k - 1
    b = (k == K) ? 1 : k + 1

    Ya = Ys[a]
    Yk = Ys[k]
    Yb = Ys[b]

    _Zk11 = Ec3(a, 2, zdims) * _Ya * Ec3(a, 2, zdims)' # Ya[2,2]
    _Zk12 = Ec3(k, 1, zdims) * _Yk * Ec3(k, 2, zdims)' # Yk[1,2]
    _Zk13 = Ec3(k, 1, zdims) * _Yk * Ec3(k, 3, zdims)' # Yk[1,3]
    _Zk22 = Ec3(b, 1, zdims) * _Yb * Ec3(b, 1, zdims)' # Yb[1,1]
    _Zk23 = Ec3(k, 2, zdims) * _Yk * Ec3(k, 3, zdims)' # Yk[2,3]
    _Zk33 = Ec3(k, 3, zdims) * _Yk * Ec3(k, 3, zdims)' # Yk[3,3]

    Zk = [_Zk11 _Zk12 _Zk13; _Zk12' _Zk22 _Zk23; _Zk13' _Zk23' _Zk33]
    =#



    #=
    lenωk = length(ωs[k])
    # Jk
    _zka = zk(k, zeros(lenωk), γdims, zdims, input, safety, ffnet)
    _Jk = hcat([zk(k, e(j, lenωk), γdims, zdims, input, safety, ffnet) - _zka for j in 1:lenωk]...)
    push!(Js, _Jk)

    # Jk' * zka
    _Jktza = _Jk' * _zka
    push!(Jtzas, _Jktza)

    # inv(I + Jk' * Jk)
    _I_JktJk = Symmetric(I + _Jk' * _Jk)
    _I_JktJk_inv = inv(_I_JktJk)
    push!(I_JtJ_invs, _I_JktJk_inv)
    =#

    #
  end

  cache = (Js, Jtzas, I_JtJ_invs)
  return cache
end

# The vectorized version
function zk(k, ωk, γdims, zdims, input, safety, ffnet)
  _Zk = Zk(k, ωk, γdims, zdims, input, safety, ffnet)
  return _Zk[:]
end

# Project onto the non-negative orthant
function projectΓ(params)
  γ, _, _, _, _, _, _ = params
  return max.(abs.(γ), 0)
end

# The γ update
function stepγ(ωs, μs, ρ, γdims)
  K = length(γdims)
  tmp = [Hc(k, γdims)' * (ωs[k] + (μs[k] / ρ)) for k in 1:K]
  # tmp = dinv .* sum(tmp)
  tmp = sum(tmp) / 3 # D = 3I
  return projectΓ(tmp)
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
function stepvk(k, ωk, λk, ρ, γdims, zdims, input, safety, ffnet)
  tmp = zk(k, ωk, γdims, zdims, input, safety, ffnet)
  tmp = tmp - (λk / ρ)
  return projectNsd(tmp)
end

# The ωk update
function stepωk(k, γ, vk, λk, μk, ρ, γdims, Jk, Jtzak, I_JtJ_invk)
  tmp = Jk' * vk
  tmp = tmp + Hc(k, γdims) * γ
  tmp = tmp + (Jk' * λk - μk) / ρ
  tmp = tmp - Jtzak
  tmp = I_JtJ_invk * tmp
  return tmp
end

# The λk update
function stepλk(vk, λk, ρ, zkωk)
  tmp = λk + ρ * (vk - zkωk)
  return tmp
end

# The μk update
function stepμk(k, γ, ωk, μk, ρ, γdims)
  tmp = μk + ρ * (ωk - Hc(k, γdims) * γ)
  return tmp
end

#
function stepX(vs, ωs, λs, μs, ρ, γdims, zdims, input, safety, ffnet)
  new_γ = stepγ(ωs, μs, ρ, γdims)
  new_vs = Vector([stepvk(k, ωs[k], λs[k], ρ, γdims, zdims, input, safety, ffnet) for k in 1:ffnet.K])
  return (new_γ, new_vs)
end

#
function stepY(γ, vs, λs, μs, ρ, γdims, Js, Jtzas, I_JtJ_invs)
  K = length(γdims)
  new_ωs = Vector([stepωk(k, γ, vs[k], λs[k], μs[k], ρ, γdims, Js[k], Jtzas[k], I_JtJ_invs[k]) for k in 1:K])
  return new_ωs
end

#
function stepZ(γ, vs, ωs, λs, μs, ρ, γdims, zdims, input, safety, ffnet)
  K = length(γdims)
  new_λs = Vector([stepλk(vs[k], λs[k], ρ,
                          zk(k, ωs[k], γdims, zdims, input, safety, ffnet))
                    for k in 1:K])
  new_μs = Vector([stepμk(k, γ, ωs[k], μs[k], ρ, γdims) for k in 1:K])
  return (new_λs, new_μs)
end

#
function stopcond(γ, vs, ωs, λs, μs)
  return True
end


#
function admm(γ, vs, ωs, λs, μs, ρ, Js, Jtzas, I_JtJ_invs, γdims, zdims, input, safety, ffnet)

  (γ_0, vs_0, ωs_0, λs_0, μs_0) = (γ, vs, ωs, λs, μs)

  T = 10

  for t = 1:T
    step_start_time = time()
    (γ_1, vs_1) = stepX(vs_0, ωs_0, λs_0, μs_0, ρ, γdims, zdims, input, safety, ffnet)
    ωs_1 = stepY(γ_1, vs_1, λs_0, μs_0, ρ, γdims, Js, Jtzas, I_JtJ_invs)
    (λs_1, μs_1) = stepZ(γ_1, vs_1, ωs_1, λs_0, μs_0, ρ, γdims, zdims, input, safety, ffnet)

    #
    (γ_0, vs_0, ωs_0, λs_0, μs_0) = (γ_1, vs_1, ωs_1, λs_1, μs_1)

    #
    step_total_time = time() - step_start_time
    println("ADMM step t = " * string(t) * "/" * string(T) * ", time = " * string(step_total_time))
  end

  return (γ_0, vs_0, ωs_0, λs_0, μs_0)
end


function run()
end

export stepω

end # End module
