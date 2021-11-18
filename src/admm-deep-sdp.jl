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
  Hcinds :: Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Tuple{Int, Int}}} # γa, γk, γb (start,end)
end

# Initialize this structure to customize options
@with_kw struct AdmmOptions
  max_iters :: Int = 40
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: Int = 2
  nsd_tol :: Float64 = 1e-4
  ρ :: Float64 = 1.0
  verbose :: Bool = false
end

# Initialize zero parameters of the appropriate size
function initParams(inst :: VerificationInstance, opts :: AdmmOptions)
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

  ρ = opts.ρ

  params = AdmmParams(γ=γ, vs=vs, ωs=ωs, λs=λs, μs=μs, ρ=ρ, γdims=γdims, zdims=zdims)
  return params
end

# Computes some potentially expensive things
function precompute(params :: AdmmParams, inst :: VerificationInstance, opts :: AdmmOptions)
  @assert inst.net.nettype isa ReluNetwork # Only handle this for now
  @assert params.K == inst.net.K

  if opts.verbose
    println("precompute: start!" )
  end

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
    iterk_start_time = time()

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

    iterk_time = time() - iterk_start_time
    if opts.verbose
      println("precompute: Yss[" * string(k) * "/" * string(K) * "], time: " * string(iterk_time))
    end
  end

  Js = Vector{Any}()
  zaffs = Vector{Any}()
  Jtzaffs = Vector{Any}()
  I_JtJ_invs = Vector{Any}()

  # Populate the Js and its dependencies
  for k = 1:K
    iterk_start_time = time()

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

    iterk_time = time() - iterk_start_time
    if opts.verbose
      println("precompute: Js[" * string(k) * "/" * string(K) * "], time: " * string(iterk_time))
    end
  end

  # Calculate the Hcinds
  Hcinds = Vector{Any}()
  for k = 1:K
    a = (k == 1) ? K : k - 1
    b = (k == K) ? 1 : k + 1
    alow = sum(γdims[1:a-1]) + 1
    ahigh = sum(γdims[1:a])
    klow = sum(γdims[1:k-1]) + 1
    khigh = sum(γdims[1:k])
    blow = sum(γdims[1:b-1]) + 1
    bhigh = sum(γdims[1:b])
    kinds = ((alow, ahigh), (klow, khigh), (blow, bhigh))
    push!(Hcinds, kinds)
  end

  cache = AdmmCache(Js=Js, zaffs=zaffs, Jtzaffs=Jtzaffs, I_JtJ_invs=I_JtJ_invs, Hcinds=Hcinds)
  return cache
end

# Fast Hc access
function cachedHcSelect(k, γ, cache :: AdmmCache)
  (alow, ahigh), (klow, khigh), (blow, bhigh) = cache.Hcinds[k]
  γa = γ[alow:ahigh]
  γk = γ[klow:khigh]
  γb = γ[blow:bhigh]
  return [γa; γk; γb]
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
  tmp = Symmetric(reshape(vk, (dim, dim)))
  eig = eigen(tmp)
  tmp = Symmetric(eig.vectors * Diagonal(min.(eig.values, 0)) * eig.vectors')
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
  tmp = tmp + cachedHcSelect(k, params.γ, cache)
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
  tmp = params.ωs[k] - cachedHcSelect(k, params.γ, cache)
  tmp = params.μs[k] + params.ρ * tmp
  return tmp
end

# The X = {γ, v1, ..., vK} variable updates
function stepX(params :: AdmmParams, cache :: AdmmCache)
  new_γ = stepγ(params, cache)
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])
  return (new_γ, new_vs)
end

# The X = {γ, v1, ..., vK} variable updates, but with some Newton's method
function stepXNewton(params :: AdmmParams, cache :: AdmmCache)
  # Pretend that the objective function is
  #    f(γ) = (1/2) ∑ || Hk γ - ωk - μk/ρ ||^2
  #   ∇f(γ) = ∑ Hk' (Hk γ - ωk - μk/ρ) = D γ - ∑Hk'(ωk + μk/ρ)
  #  ∇2f(γ) = ∑ Hk' Hk = D = 3 I

  γt = stepγ(params, cache)
  α = 0.5 # step size
  T = 3
  for t = 1:T
    oldγt = γt
    ∇f = 3*γt - sum(Hc(k, params.γdims)' * (params.ωs[k] + (params.μs[k] / params.ρ)) for k in 1:params.K)
    γt = γt - (α / 3) * ∇f
    γt = projectΓ(γt)
  end
  new_γ = γt
  new_vs = Vector([stepvk(k, params, cache) for k in 1:params.K])
  return (new_γ, new_vs)
end

# The Y = {ω1, ..., ωK} variable updates
function stepY(params :: AdmmParams, cache :: AdmmCache)
  new_ωs = Vector([stepωk(k, params, cache) for k in 1:params.K])
  return new_ωs
end

# The Z = {λ1, ..., λK, μ1, ..., μK} variable updates
function stepZ(params :: AdmmParams, cache :: AdmmCache)
  new_λs = Vector([stepλk(k, params, cache) for k in 1:params.K])
  new_μs = Vector([stepμk(k, params, cache) for k in 1:params.K])
  return (new_λs, new_μs)
end

# Check that each Qk <= 0
function isγSat(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmOptions)
  for k in 1:params.K
    ωk = cachedHcSelect(k, params.γ, cache)
    _zk = zk(k, ωk, cache)
    dim = Int(round(sqrt(length(_zk))))
    tmp = Symmetric(reshape(_zk, (dim, dim)))
    eig = eigen(tmp)

    if maximum(eig.values) > opts.nsd_tol
      return false
    end
  end
  return true
end

# Should we be stopping?
function shouldStop(t, params :: AdmmParams, cache :: AdmmCache, opts :: AdmmOptions)
  if t > opts.begin_check_at_iter && mod(t, opts.check_every_k_iters) == 0
    if isγSat(params, cache, opts)
      if opts.verbose
        println("Sat!")
      end
      return true
    end
  end
  return false
end

#
function admm(params :: AdmmParams, cache :: AdmmCache, opts :: AdmmOptions)
  iter_params = deepcopy(params)
  iters_run = 0
  T = opts.max_iters
  total_time = 0
  for t = 1:T
    step_start_time = time()

    xstart_time = time()
    new_γ, new_vs = stepX(iter_params, cache)
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

    # Coalesce all the time statistics
    step_total_time = time() - step_start_time
    total_time = total_time + step_total_time
    all_times = (xtotal_time, ytotal_time, ztotal_time, step_total_time, total_time)
    if opts.verbose
      println("step[" * string(t) * "/" * string(T) * "]" * " time: " * string(round.(all_times, digits=1)))
    end

    iters_run = t

    if shouldStop(t, iter_params, cache, opts)
      break
    end
  end

  return iter_params, isγSat(iter_params, cache, opts), iters_run, total_time
end

# Call this
function run(inst :: VerificationInstance, opts :: AdmmOptions)
  start_time = time()
  start_params = initParams(inst, opts)

  precompute_start_time = time()
  cache = precompute(start_params, inst, opts)
  precompute_time = time() - precompute_start_time

  new_params, issat, iters_run, admm_iters_time = admm(start_params, cache, opts)
  total_time = time() - start_time
  output = SolutionOutput(
            model = new_params,
            summary = "precompute time: " * string(precompute_time) * ", iters run: " * string(iters_run),
            status = issat ? "OPTIMAL" : "UNKNOWN",
            total_time = total_time,
            solve_time = admm_iters_time)
  return output
end

export AdmmOptions
export run

end # End module

