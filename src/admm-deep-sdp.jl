# Implement ADMM
module AdmmDeepSdp

using ..Header
using ..Common
using ..SplitDeepSdpB: Zk # Helpful to re-use
using Parameters
using LinearAlgebra


@with_kw mutable struct AdmmParams
  γ; vs; ωs; λs; μs;
  ρ :: Float64;
  γdims :: Vector{Int};
  K :: Int = length(γdims)
end

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
  for k = 1:K; push!(vs, zeros(length(vec(Zk(k, ωs[k], γdims, zdims, inst.input, inst.safety, ffnet))))) end

  # λk are the dual vars for vk = zk(ωk)
  λs = [zeros(length(vs[k])) for k in 1:K]

  # μk are the dual vars for ωk = Hk * γ
  μs = [zeros(length(ωs[k])) for k in 1:K]

  # A guess
  ρ = 1.0

  params = AdmmParams(γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs, ρ=ρ, γdims=γdims)
  return params
end

# Computes some potentially expensive things
function precompute(params, inst)
  @assert inst.net.nettype isa ReluNetwork # Only handle this for now
  @assert params.K == inst.net.K

  println("precompute: start!" )

  input = inst.input
  safety = inst.safety
  ffnet = inst.net
  K = ffnet.K
  zdims = [ffnet.xdims[1:K]; 1]
  γdims = params.γdims

  # Each Yss[k] contains the non-affine components of Yk
  Yss = Vector{Any}()

  # Contains the affine components of each Yk
  Yaffs = Vector{Any}()

  # Calculate the Ys parts
  for k = 1:K
    kstart_time = time()

    Ykaff = (k == K) ? YγK(zeros(γdims[k]), input, safety, ffnet) : Yγk(k, zeros(γdims[k]), ffnet)
    Ykparts = Vector{Any}()
    for j = 1:γdims[k]
      tmp = (k == K) ? YγK(e(j, γdims[k]), input, safety, ffnet) : Yγk(k, e(j, γdims[k]), ffnet)
      Ykj = tmp - Ykaff
      push!(Ykparts, Ykj)
    end
    # Push Ykparts, and account for the affine component separately
    push!(Yss, Ykparts)
    push!(Yaffs, Ykaff)

    ktotal_time = time() - kstart_time
    println("precompute: Yss k = " * string(k) * "/" * string(K) * ", time = " * string(ktotal_time))
  end

  Js = Vector{Any}()
  Jtzaffs = Vector{Any}()
  I_JtJ_invs = Vector{Any}()

  # Populate the Js and its dependencies
  for k = 1:K
    kstart_time = time()

    a = (k == 1) ? K : k - 1
    b = (k == K) ? 1 : k + 1

    Yaparts = Yss[a]
    Ykparts = Yss[k]
    Ybparts = Yss[b]

    Jk = Vector{Any}()

    # The strategy is to decompose each Zk = aug(Zk[1,1]) + ... + aug(Zk[3,3]),
    # where aug is an embiggen op such that size(Zk) == size(aug(Zk[i,j]))
    # and then compute the γ-wise contribution of each component to build
    # the Jacobian directly.

    # Cache some selectors
    Fk1t_Fa2 = Ec3(k, 1, zdims)' * Ec3(a, 2, zdims)
    Fa2t_Fk1 = Ec3(a, 2, zdims)' * Ec3(k, 1, zdims)
    Fk1t_Fk1 = Ec3(k, 1, zdims)' * Ec3(k, 1, zdims)
    Fk2t_Fk2 = Ec3(k, 2, zdims)' * Ec3(k, 2, zdims)
    Fk3t_Fk3 = Ec3(k, 3, zdims)' * Ec3(k, 3, zdims)
    Fk2t_Fb1 = Ec3(k, 2, zdims)' * Ec3(b, 1, zdims)
    Fb1t_Fk2 = Ec3(b, 1, zdims)' * Ec3(k, 2, zdims)

    # The Ya components first
    # aug((Yaj)[2,2]) ~ (F1' * (F2 * Yaj * F2') * F1)
    _Jka = hcat([
        let aug_Zkj11 = Fk1t_Fa2 * Yaparts[j] * Fa2t_Fk1;
          vec(aug_Zkj11) end
        for j in 1:γdims[a]]...)

    # The Yk components next
    # aug((Ykj)[1,2]) ~ (F1' * (F1 * Ykj * F2') * F2) ~ aug((Ykj)[2,1])'
    # aug((Ykj)[1,3]) ~ (F1' * (F1 * Ykj * F3') * F3) ~ aug((Ykj)[3,1])'
    # aug((Ykj)[2,3]) ~ (F2' * (F2 * Ykj * F3') * F3) ~ aug((Ykj)[3,2])'
    # aug((Ykj)[3,3]) ~ (F3' * (F3 * Ykj * F3') * F3)
    _Jkk = hcat([
        let aug_Zkj12 = Fk1t_Fk1 * Ykparts[j] * Fk2t_Fk2,
            aug_Zkj13 = Fk1t_Fk1 * Ykparts[j] * Fk3t_Fk3,
            aug_Zkj23 = Fk2t_Fk2 * Ykparts[j] * Fk3t_Fk3,
            aug_Zkj33 = Fk3t_Fk3 * Ykparts[j] * Fk3t_Fk3,
            tmp = aug_Zkj12 + aug_Zkj13 + aug_Zkj23,
            tmp = tmp + tmp',
            tmp = tmp + aug_Zkj33;
          vec(tmp) end
        for j in 1:γdims[k]]...)

    # The Yb components last
    _Jkb = hcat([
        let aug_Zkj22 = Fk2t_Fb1 * Ybparts[j] * Fb1t_Fk2;
          vec(aug_Zkj22) end
        for j in 1:γdims[b]]...)

    Jk = [_Jka _Jkk _Jkb]

    push!(Js, Jk)
    
    # Calculate the affine component of Zk
    aug_Za_aff11 = Fk1t_Fa2 * Yaffs[a] * Fa2t_Fk1
    aug_Zk_aff12 = Fk1t_Fk1 * Yaffs[k] * Fk2t_Fk2
    aug_Zk_aff13 = Fk1t_Fk1 * Yaffs[k] * Fk3t_Fk3
    aug_Zk_aff23 = Fk2t_Fk2 * Yaffs[k] * Fk3t_Fk3
    aug_Zk_aff33 = Fk3t_Fk3 * Yaffs[k] * Fk3t_Fk3
    aff_tmp = aug_Zk_aff12 + aug_Zk_aff13 + aug_Zk_aff23
    aff_tmp = aff_tmp + aff_tmp'
    aff_tmp = aff_tmp + aug_Zk_aff33
    aug_Za_aff22 = Fk2t_Fb1 * Yaffs[b] * Fb1t_Fk2
    Zkaff = aug_Za_aff11 + aff_tmp + aug_Za_aff22
    zkaff = vec(Zkaff)

    # Jk' * zkaff
    _Jktzaff = Jk' * zkaff
    push!(Jtzaffs, _Jktzaff)

    # inv(I + Jk' * Jk)
    _I_JktJk_inv = inv(Symmetric(I + Jk' * Jk))
    push!(I_JtJ_invs, _I_JktJk_inv)

    ktotal_time = time() - kstart_time
    println("precompute: Js k = " * string(k) * "/" * string(K) * ", time = " * string(ktotal_time))
  end

  cache = (Js, Jtzaffs, I_JtJ_invs)
  return cache
end

# Project onto the non-negative orthant
function projectΓ(γ)
  return max.(abs.(γ), 0)
end

# The γ update
function stepγ(params, cache)
  ωs = params.ωs
  μs = params.μs
  ρ = params.ρ
  γdims = params.γdims
  K = params.K
  tmp = [Hc(k, γdims)' * (ωs[k] + (μs[k] / ρ)) for k in 1:K]
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
function stepvk(k, params, cache)
  ωk = params.ωs[k]
  λk = params.λs[k]
  ρ = params.ρ
  Js, _, _ = cache
  tmp = Js[k] * ωk
  tmp = tmp - (λk / ρ)
  return projectNsd(tmp)
end

# The ωk update
# function stepωk(k, γ, vk, λk, μk, ρ, γdims, Jk, Jtzak, I_JtJ_invk)
function stepωk(k, params, cache)
  γ = params.γ
  vs = params.vs
  λs = params.λs
  μs = params.μs
  ρ = params.ρ
  γdims = params.γdims
  Js, Jtzaffs, I_JtJ_invs = cache
  tmp = Js[k]' * vs[k]
  tmp = tmp + Hc(k, γdims) * γ
  tmp = tmp + (Js[k]' * λs[k] - μs[k]) / ρ
  tmp = tmp - Jtzaffs[k]
  tmp = I_JtJ_invs[k] * tmp
  return tmp
end

# The λk update
function stepλk(k, params, cache)
  vs = params.vs
  ωs = params.ωs
  λs = params.λs
  ρ = params.ρ
  Js, _, _ = cache
  tmp = vs[k] - Js[k] * ωs[k]
  tmp = λs[k] + ρ * tmp
  return tmp
end

# The μk update
function stepμk(k, params, cache)
  γ = params.γ
  ωs = params.ωs
  μs = params.μs
  ρ = params.ρ
  γdims = params.γdims
  tmp = μs[k] + ρ * (ωs[k] - Hc(k, γdims) * γ)
  return tmp
end

#
function stepX(params, cache)
  new_γ = stepγ(params, cache)
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])
  return (new_γ, new_vs)
end

#
function stepY(params, cache)
  new_ωs = Vector([stepωk(k, params, cache) for k in 1:params.K])
  return new_ωs
end

#
function stepZ(params, cache)
  new_λs = Vector([stepλk(k, params, cache) for k in 1:params.K])
  new_μs = Vector([stepμk(k, params, cache) for k in 1:params.K])
  return (new_λs, new_μs)
end

#
function stopcond(γ, vs, ωs, λs, μs)
  return True
end


#
function admm(params, cache)

  iter_params = AdmmParams(
        γ = params.γ,
        vs = params.vs,
        ωs = params.ωs,
        λs = params.λs,
        μs = params.μs,
        γdims = params.γdims,
        ρ = params.ρ)

  T = 10

  for t = 1:T
    step_start_time = time()

    (new_γ, new_vs) = stepX(iter_params, cache)
    iter_params.γ = new_γ
    iter_params.vs = new_vs

    new_ωs = stepY(iter_params, cache)
    iter_params.ωs = new_ωs

    (new_λs, new_μs) = stepZ(iter_params, cache)
    iter_params.λs = new_λs
    iter_params.μs = new_μs

    #
    step_total_time = time() - step_start_time
    println("ADMM step t = " * string(t) * "/" * string(T) * ", time = " * string(step_total_time))
  end

  return iter_params
end


function run()
end

export stepω

end # End module
