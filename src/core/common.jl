# Some functionalities that will be common across different algorithms
module Common

using ..Header
using Parameters
using LinearAlgebra
using JuMP

# The ith basis vector
function e(i :: Int, dim :: Int)
  @assert 1 <= i <= dim
  e = zeros(dim)
  e[i] = 1
  return e
end

# The ith block index matrix
function E(i :: Int, dims :: Vector{Int})
  @assert 1 <= i <= length(dims)
  width = sum(dims)
  low = sum(dims[1:i-1]) + 1
  high = sum(dims[1:i])
  E = zeros(dims[i], width)
  E[1:dims[i], low:high] = I(dims[i])
  return E
end

# The block index matrix that is [E(k, dims), ..., E(k+b, dims)]
function E(k :: Int, b :: Int, dims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(dims)
  Eks = [E(k+j, dims) for j in 0:b]
  return vcat(Eks...)
end

# The clique Ck = {x[k], ..., x[k+b], x[K], x[aff]}
# For these cliques we enforce b >= 1
# Each clique has minimum size
function Ec(k :: Int, b :: Int, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(zdims) - 2 # So to exclude K and affine
  @assert zdims[end] == 1 # Affine component
  Ekb = E(k, b, zdims)
  EK = E(length(zdims)-1, zdims)
  Ea = E(length(zdims), zdims)
  return [Ekb; EK; Ea]
end

# The dimension components of a clique
function Cdims(k :: Int, b :: Int, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(zdims) - 2 # So to exclude K and affine
  @assert zdims[end] == 1
  dims = [zdims[k:k+b]; zdims[end-1]; zdims[end]]
  return dims
end

# Make the Ack matrix, for 1 <= k + b <= K, to include all the transitions
function makeAc(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  edims = ffnet.zdims[k:k+b]
  fdims = edims[2:end]
  Ack = sum(E(j, fdims)' * ffnet.Ms[k+j-1][1:end, 1:end-1] * E(j, edims) for j in 1:b)
  return Ack
end

# Make the bck stacked vector
function makebc(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  bck = vcat([ffnet.Ms[k+j-1][1:end, end] for j in 1:b]...)
  return bck
end

# Make the Bck matrix
function makeBc(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  edims = ffnet.zdims[k:k+b]
  fdims = edims[2:end]
  Bck = sum(E(j, fdims)' * E(j+1, edims) for j in 1:b)
  return Bck
end

# P function for a box
function makePbox(x1min :: Vector{Float64}, x1max :: Vector{Float64}, γ)
  @assert length(x1min) == length(x1max) == length(γ)
  Γ = Diagonal(γ)
  _P11 = -2 * Γ
  _P12 = Γ * (x1min + x1max)
  _P22 = -2 * x1min' * Γ * x1max
  P = [_P11 _P12; _P12' _P22]
  return P
end

# P function for a polytope
function makePpolytope(H :: Matrix{Float64}, h :: Vector{Float64}, Γ)
  @assert size(H)[1] == length(h)
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

# Bounding hyperplane such that normal' * f(x) <= h, for variable h
function makeShyperplane(normal :: Vector{Float64}, h, ffnet :: FeedForwardNetwork)
  d1 = ffnet.xdims[1]
  dK1 = ffnet.xdims[end]
  @assert length(normal) == dK1
  _S11 = zeros(d1, d1)
  _S12 = zeros(d1, dK1)
  _S13 = zeros(d1)
  _S22 = zeros(dK1, dK1)
  _S23 = normal
  _S33 = -2 * h
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end
#
@with_kw struct Xqinfo
  ffnet :: FeedForwardNetwork
  ϕout_intv :: Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}} = nothing
  slope_intv :: Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}} = nothing
  tband :: Int; @assert tband >= 0
end

# Calculate how large λ should be given a tband
function λlength(qxdim :: Int, tband :: Int)
  @assert 0 <= tband <= qxdim - 1
  return sum((qxdim-tband):qxdim)
end

# Make the diagλ and T matrices
function makeDiagλandT(qxdim :: Int, λ, tband :: Int)
  tstart = time()
  @assert length(λ) == λlength(qxdim, tband)
  diagλ = λ[1:qxdim]


  println("qxdim: " * string(qxdim))

  # Given a pair i,j, calculate its relative index in the λ vector
  pair2ind(i,j) = sum((qxdim-(j-i)+1):qxdim) + i

  if tband > 0
    ijs = [(i, j) for i in 1:(qxdim-1) for j in (i+1):qxdim if j-i <= tband]
    δts = [e(i, qxdim)' - e(j, qxdim)' for (i, j) in ijs]
    Δ = vcat(δts...)
    v = vec([λ[pair2ind(i,j)] for (i,j) in ijs])
    T = Δ' * (v .* Δ)
  else
    T = zeros(qxdim, qxdim)
  end

  println("total time making T: " * string(time() - tstart))
  return diagλ, T
end

#=
# Make a T matrix of a particular size given matrix of variables τ
function makeT(dim :: Int, τ)
  @assert dim >= 1
  @assert size(τ) == (dim, dim)
  ijs = [(i, j) for i in 1:(dim-1) for j in (i+1):dim]
  δts = [e(i, dim)' - e(j, dim)' for (i, j) in ijs]
  Δ = vcat(δts...)
  v = vec([τ[i,j] for (i, j) in ijs])

  tstart = time()
  T = Δ' * (v .* Δ)
  println("mult total tme: " * string(time() - tstart))

  #=
  V = diagm(vec([τ[i,j] for (i, j) in ijs]))
  println("size Δ: "  * string(size(Δ)))
  println("size V: "  * string(size(V)))
  T = Δ' * V * Δ
  =#
  println("makeT: F")
  return T
end
=#

function makeQrelu(qxdim :: Int, λ, η, ν, xqinfo :: Xqinfo)
  @assert length(λ) == λlength(qxdim, xqinfo.tband)
  ε = 1e-6
  ϕa, ϕb = xqinfo.slope_intv
  @assert length(ϕa) == length(ϕb)
  @assert -ε <= minimum(ϕa) && maximum(ϕa) <= 1 + ε
  @assert -ε <= minimum(ϕb) && maximum(ϕb) <= 1 + ε

  diagλ, T = makeDiagλandT(qxdim, λ, xqinfo.tband)
  s0, s1 = 0, 1
  _Q11 = -2 * diagm(ϕa .* ϕb .* diagλ) - 2 * (s0 * s1 * T)
  _Q12 = diagm((ϕa + ϕb) .* diagλ) + (s0 + s1) * T
  _Q13 = -(ϕa .* η) - (ϕb.* ν)
  _Q22 = -2 * T
  _Q23 = η + ν
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

# Make Q with help of the ϕ(x) bounds
function makeQϕout(qxdim :: Int, d, xqinfo :: Xqinfo)
  ϕmin, ϕmax = xqinfo.ϕout_intv
  @assert all(ϕmin .<= ϕmax)
  @assert -Inf < minimum(ϕmin) && maximum(ϕmax) < Inf
  @assert length(d) == length(ϕmin) == length(ϕmax) == qxdim

  D = diagm(d)
  _Q11 = zeros(qxdim, qxdim)
  _Q12 = zeros(qxdim, qxdim)
  _Q13 = zeros(qxdim)
  _Q22 = -2 * D
  _Q23 = D * (ϕmin + ϕmax)
  _Q33 = -2 * ϕmin' * D * ϕmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end


# Make an Xk, with help from the Xqinfo
function makeXq(k :: Int, b :: Int, vars, xqinfo :: Xqinfo)
  ffnet = xqinfo.ffnet
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  qxdim = sum(ffnet.zdims[k+1:k+b])

  if ffnet.type isa ReluNetwork
    λ_slope, η_slope, ν_slope, d_out = vars
    @assert length(λ_slope) == λlength(qxdim, xqinfo.tband)
    @assert length(η_slope) == length(ν_slope) == qxdim
    @assert length(d_out) == qxdim

    Q = makeQrelu(qxdim, λ_slope, η_slope, ν_slope, xqinfo)

    # If the ϕ intervals exist and are not infinite, use them
    if (!(xqinfo.ϕout_intv isa Nothing)
        && (-Inf < minimum(xqinfo.ϕout_intv[1]) && maximum(xqinfo.ϕout_intv[2]) < Inf))
      Q = Q + makeQϕout(qxdim, d_out, xqinfo)
    end

  # Other kinds
  else
    error("unsupported network: " * string(ffnet.type))
  end

  _R11 = makeAc(k, b, ffnet)
  _R12 = makebc(k, b, ffnet)
  _R21 = makeBc(k, b, ffnet)
  _R22 = zeros(size(_R12))
  _R31 = zeros(1, size(_R21)[2])
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  return R' * Q * R
end

# γ is a vector
function makeXin(γ, input :: InputConstraint, ffnet :: FeedForwardNetwork)
  d1 = ffnet.xdims[1]
  if input isa BoxInput
    @assert length(γ) == d1
    return makePbox(input.x1min, input.x1max, γ)
  elseif input isa PolytopeInput
    @assert length(γ) == d1^2
    Γ = reshape(γ, (d1, d1))
    return makePpolytope(input.H, input.h, Γ)
  else
    error("unsupported input constraints: " * string(input))
  end
end

# Make a safety constraint wrt a fixed S matrix
function makeXsafe(S, ffnet :: FeedForwardNetwork)
  WK = ffnet.Ms[ffnet.K][1:end, 1:end-1]
  bK = ffnet.Ms[ffnet.K][1:end, end]

  d1 = ffnet.zdims[1]
  (dK1, dK) = size(WK)

  _R11 = I(d1)
  _R12 = zeros(d1, dK)
  _R13 = zeros(d1)
  _R21 = zeros(dK1, d1)
  _R22 = WK
  _R23 = bK
  _R31 = zeros(1, d1)
  _R32 = zeros(1, dK)
  _R33 = 1
  R = [_R11 _R12 _R13; _R21 _R22 _R23; _R31 _R32 _R33]
  return R' * S * R
end

# Slowly using up the English alphabet
export e, E, Ec, Cdims
export makeAc, makebc, makeBc
export makePbox, makePpolytope
export makeShyperplane
export λlength
export makeQϕout, makeQrelu
export Xqinfo
export makeXq, makeXin, makeXsafe

end # End module

