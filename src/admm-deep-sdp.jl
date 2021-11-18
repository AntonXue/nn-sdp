# Implement ADMM
module AdmmDeepSdp

using ..Header
using ..Common
using ..SplitDeepSdpB: Zk # Helpful to re-use
using Parameters
using LinearAlgebra
using JuMP
using Mosek
using MosekTools

# Track the parameters that are updated at each ADMM tsep
@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ωs :: Vector{Vector{Float64}}
  λs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}
  ρ :: Float64
  γdims :: Vector{Int}
  zdims :: Vector{Int}
  K :: Int = length(γdims)
end

# Precompute stuff
@with_kw struct AdmmCache
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
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
  ρ = 10.0

  params = AdmmParams(γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs, ρ=ρ, γdims=γdims, zdims=zdims)
  return params
end

# Computes some potentially expensive things
function precompute(params :: AdmmParams, inst :: VerificationInstance)
  @assert inst.net.nettype isa ReluNetwork # Only handle this for now
  @assert params.K == inst.net.K

  println("precompute: start!" )

  input = inst.input
  safety = inst.safety
  ffnet = inst.net
  K = ffnet.K
  zdims = params.zdims
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
  zaffs = Vector{Any}()
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
    push!(zaffs, zkaff)

    # Jk' * zkaff
    _Jktzaff = Jk' * zkaff
    push!(Jtzaffs, _Jktzaff)

    # inv(I + Jk' * Jk)
    _I_JktJk_inv = inv(Symmetric(I + Jk' * Jk))
    push!(I_JtJ_invs, _I_JktJk_inv)

    ktotal_time = time() - kstart_time
    println("precompute: Js k = " * string(k) * "/" * string(K) * ", time = " * string(ktotal_time))
  end

  cache = AdmmCache(Js=Js, zaffs=zaffs, Jtzaffs=Jtzaffs, I_JtJ_invs=I_JtJ_invs)
  return cache
end

# Calculate the vectorized Zk
function zk(k :: Int, ωk :: Vector{Float64}, cache :: AdmmCache)
  return cache.Js[k] * ωk + cache.zaffs[k]
end

# Project onto the non-negative orthant
function projectΓ(γ :: Vector{Float64})
  return max.(γ, 0)
end

# The γ update
function stepγ(params :: AdmmParams, cache :: AdmmCache)
  tmp = [Hc(k, params.γdims)' * (params.ωs[k] + (params.μs[k] / params.ρ)) for k in 1:params.K]
  tmp = sum(tmp) / 3 # D = 3I
  return projectΓ(tmp)
end

# Project a vector onto the negative semidefinite cone
function projectNsd(vk :: Vector{Float64})
  dim = Int(round(sqrt(length(vk)))) # :)
  @assert length(vk) == dim * dim
  tmp = reshape(vk, (dim, dim))
  eig = eigen(tmp)
  # tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
  tmp = Symmetric(eig.vectors * Diagonal(min.(real.(eig.values), 0)) * eig.vectors')
  return tmp[:]
end

# The vk update
function stepvk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = zk(k, params.ωs[k], cache)
  tmp = tmp - (params.λs[k] / params.ρ)
  return projectNsd(tmp)
end

# The ωk update
function stepωk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = cache.Js[k]' * params.vs[k]
  tmp = tmp + Hc(k, params.γdims) * params.γ
  tmp = tmp + (cache.Js[k]' * params.λs[k] - params.μs[k]) / params.ρ
  tmp = tmp - cache.Jtzaffs[k]
  tmp = cache.I_JtJ_invs[k] * tmp
  return tmp
end

# The λk update
function stepλk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.vs[k] - zk(k, params.ωs[k], cache)
  tmp = params.λs[k] + params.ρ * tmp
  return tmp
end

# The μk update
function stepμk(k :: Int, params :: AdmmParams, cache :: AdmmCache)
  tmp = params.ωs[k] - Hc(k, params.γdims) * params.γ
  tmp = params.μs[k] + params.ρ * tmp
  return tmp
end

#
function stepX(params :: AdmmParams, cache :: AdmmCache)
  new_γ = stepγ(params, cache)
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])
  return (new_γ, new_vs)
end

function stepXsolveBoth(params :: AdmmParams, cache :: AdmmCache)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9
  ))
  @variable(model, var_γ[1:length(params.γ)] >= 0)
  var_Vs = Vector{Any}()
  for k in 1:params.K
    vkdim = Int(round(sqrt(length(params.vs[k]))))
    var_Vk = @variable(model, [1:vkdim, 1:vkdim])
    @SDconstraint(model, var_Vk <= 0)
    push!(var_Vs, var_Vk)
  end

  vparts = Vector{Any}()
  γparts = Vector{Any}()
  for k in 1:params.K
    push!(vparts, vec(var_Vs[k]) - zk(k, params.ωs[k], cache) + (params.λs[k] / params.ρ))
    push!(γparts, params.ωs[k] - Hc(k, params.γdims) * var_γ + (params.μs[k] / params.ρ))
  end

  vsum = sum(vparts[k]' * vparts[k] for k in 1:params.K)
  γsum = sum(γparts[k]' * γparts[k] for k in 1:params.K)

  @objective(model, Min, vsum + γsum)
  optimize!(model)
  new_γ = value.(var_γ)
  new_vs = [vec(value.(var_Vs[k])) for k in 1:params.K]
  return (new_γ, new_vs)
end

function stepXsolveγOnly(params :: AdmmParams, cache :: AdmmCache)
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9
  ))
  @variable(model, var_γ[1:length(params.γ)] >= 0)

  γparts = [params.ωs[k] - Hc(k, params.γdims) * var_γ + (params.μs[k] / params.ρ) for k in 1:params.K]
  γsum = sum(γparts[k]' * γparts[k] for k in 1:params.K)
  @objective(model, Min, γsum)
  optimize!(model)

  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])
  new_γ = value.(var_γ)
  return (new_γ, real.(new_vs))
end

function stepXNewton(params :: AdmmParams, cache :: AdmmCache)
  # Pretend that the objective function is
  #    f(γ) = (1/2) ∑ || Hk γ - ωk - μk/ρ ||^2
  #   ∇f(γ) = ∑ Hk' (Hk γ - ωk - μk/ρ) = D γ - ∑Hk'(ωk + μk/ρ)
  #  ∇2f(γ) = ∑ Hk' Hk = D = 3 I

  γt = params.γ
  α = 0.5 # step size
  T = 10
  for t = 1:T
    oldγt = γt
    ∇f = 3*γt - sum(Hc(k, params.γdims)' * (params.ωs[k] + (params.μs[k] / params.ρ)) for k in 1:params.K)
    γt = γt - (α / 3) * ∇f
    γt = projectΓ(γt)
    println("\tnorm diff = " * string(norm(oldγt - γt)))
  end
  new_γ = γt
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])


  #=
  model = Model(optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
    "INTPNT_CO_TOL_DFEAS" => 1e-9
  ))
  @variable(model, var_γ[1:length(params.γ)] >= 0)

  γparts = [params.ωs[k] - Hc(k, params.γdims) * var_γ + (params.μs[k] / params.ρ) for k in 1:params.K]
  γsum = sum(γparts[k]' * γparts[k] for k in 1:params.K)
  @objective(model, Min, γsum)
  optimize!(model)
  =#

  mosekγ = value.(var_γ)
  println("\tmosek_γ diff = " * string(norm(new_γ - mosekγ)))

  return (new_γ, new_vs)
end

#
function stepY(params :: AdmmParams, cache :: AdmmCache)
  new_ωs = Vector([stepωk(k, params, cache) for k in 1:params.K])
  return new_ωs
end

#
function stepZ(params :: AdmmParams, cache :: AdmmCache)
  new_λs = Vector([stepλk(k, params, cache) for k in 1:params.K])
  new_μs = Vector([stepμk(k, params, cache) for k in 1:params.K])
  return (new_λs, new_μs)
end

#
# Check that each Qk <= 0
function isγSat(params :: AdmmParams, cache :: AdmmCache)
  for k in 1:params.K
    ωk = Hc(k, params.γdims) * params.γ
    # zk = cache.Js[k] * ωk + cache.zaffs[k] # old method
    _zk = zk(k, ωk, cache)
    dim = Int(round(sqrt(length(_zk))))
    tmp = reshape(_zk, (dim, dim))
    eig = eigen(tmp)

    if maximum(real.(eig.values)) > 1e-3
      return false
    end
  end
  return true
end


#
function stopcond(t, params, inst)
  return True
end


#
function admm(params :: AdmmParams, cache :: AdmmCache)

  iter_params = deepcopy(params)

  T = 50
  total_time = 0
  for t = 1:T
    step_start_time = time()

    xstart_time = time()
    new_γ, new_vs = stepX(iter_params, cache)
    # new_γ, new_vs = stepXsolveBoth(iter_params, cache)
    # new_γ, new_vs = stepXsolveγOnly(iter_params, cache)
    # new_γ, new_vs = stepXsolveγOnly(iter_params, cache)
    # new_γ, new_vs = stepXNewton(iter_params, cache)
    iter_params.γ = new_γ
    iter_params.vs = new_vs
    xtotal_time = time() - xstart_time

    ystart_time = time()
    new_ωs = stepY(iter_params, cache)
    iter_params.ωs = new_ωs
    ytotal_time = time() - ystart_time

    zstart_time = time()
    new_λs, new_μs = stepZ(iter_params, cache)
    iter_params.λs = new_λs
    iter_params.μs = new_μs
    ztotal_time = time() - zstart_time

    #
    step_total_time = time() - step_start_time
    total_time = total_time + step_total_time

    times = (xtotal_time, ytotal_time, ztotal_time, step_total_time, total_time)

    println("t[" * string(t) * "/" * string(T) * "]"
            * " time: " * string(round.(times, digits=1)))

    #=
    println("t[" * string(t) * "/" * string(T) * "]"
            * " time: " * string(round.(times, digits=1))
            * ", feasible: " * string(isγSat(iter_params, cache)))
    =#
    # push!(param_hist, deepcopy(iter_params))

    if t > 5 && mod(t, 3) == 0
      if isγSat(iter_params, cache)
        break
      end
    end
  end

  return iter_params
end


function run()
end

export stepω

end # End module

