# Some functionalities that will be common across different algorithms
module Common

using ..Header
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
  @assert k >= 1 && b >= 0
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

# Selectors within the clique Ck
function F(k :: Int, b :: Int, i :: Int, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1 && i >= 1
  @assert 1 <= k + b <= length(zdims) - 2 # Exclude K and affine
  @assert zdims[end] == 1 # Affine component
  Ckdims = [zdims[k:k+b]; zdims[end-1]; zdims[end]]
  return E(i, Ckdims)
end

# the K selector within the clique Ck
function FK(k :: Int, b :: Int, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(zdims) - 2 # Exclude K and affine
  @assert zdims[end] == 1 # Affine component
  Ckdims = [zdims[k:k+b]; zdims[end-1]; zdims[end]]
  return E(b+2, Ckdims)
end

# The affine selector within the clique Ck
function Faff(k :: Int, b ::Int, zdims :: Vector{Int})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(zdims) - 2 # Exclude K and affine
  @assert zdims[end] == 1 # Affine component
  Ckdims = [zdims[k:k+b]; zdims[end-1]; zdims[end]]
  return E(b+3, Ckdims)
end

# Make the A matrix
function makeA(ffnet :: FeedForwardNetwork)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  A = sum(E(j, fdims)' * ffnet.Ms[j][1:end, 1:end-1] * E(j, edims) for j in 1:ffnet.K-1)
  return A
end

# The b stacked vector
function makeb(ffnet :: FeedForwardNetwork)
  b = vcat([ffnet.Ms[j][1:end, end] for j in 1:ffnet.K-1]...)
  return b
end

# Make the B matrix
function makeB(ffnet :: FeedForwardNetwork)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:ffnet.K-1)
  return B
end

# Make the Ack matrix, for 1 <= k + b <= K, to include all the transitions
function makeAck(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  edims = ffnet.zdims[k:k+b]
  fdims = edims[2:end]
  Ack = sum(E(j, fdims)' * ffnet.Ms[k+j-1][1:end, 1:end-1] * E(j, edims) for j in 1:b)
  return Ack
end

# Make the bck stacked vector
function makebck(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  bck = vcat([ffnet.Ms[k+j-1][1:end, end] for j in 1:b]...)
  return bck
end

# Make the Bck matrix
function makeBck(k :: Int, b :: Int, ffnet :: FeedForwardNetwork)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  edims = ffnet.zdims[k:k+b]
  fdims = edims[2:end]
  Bck = sum(E(j, fdims)' * E(j+1, edims) for j in 1:b)
  return Bck
end

# A banded T matrix
function makeBandedT(Tdim :: Int, Λ, band :: Int)
  @assert size(Λ) == (Tdim, Tdim)
  T = diagm(diag(Λ))
  if band >= 1
    ijs = [(i, j) for i in 1:(Tdim-1) for j in (i+1):Tdim if abs(i-j) <= band]
    δts = [e(i, Tdim)' - e(j, Tdim)' for (i, j) in ijs]
    Δ = vcat(δts...)
    V = diagm(vec([Λ[i,j] for (i, j) in ijs]))
    T = T + Δ' * V * Δ
  end
  return T
end

# A function for making T; smaller instances Tk can be made using this
function makeT(Tdim :: Int, Λ, pattern :: TPattern)
  if pattern isa BandedPattern
    @assert size(Λ) == (Tdim, Tdim)
    return makeBandedT(Tdim, Λ, pattern.tband)
  else
    error("unsupported pattern: " * string(pattern))
  end
end

# The Q matirx
function makeQrelu(Tdim :: Int, Λ, η, ν, pattern :: TPattern; a :: Float64 = 0.0, b :: Float64 = 1.0)
  @assert size(Λ) == (Tdim, Tdim)
  @assert length(η) == length(ν) == Tdim

  T = makeT(Tdim, Λ, pattern)

  _Q11 = -2 * a * b * T
  _Q12 = (a + b) * T
  _Q13 = -(b * ν + a + η)
  _Q22 = -2 * T
  _Q23 = ν + η
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

# P function for a box
function BoxP(xbot, xtop, γ)
  @assert length(xbot) == length(xtop) == length(γ)
  Γ = Diagonal(γ)
  _P11 = -2 * Γ
  _P12 = Γ * (xbot + xtop)
  _P22 = -2 * xbot' * Γ * xtop
  P = [_P11 _P12; _P12' _P22]
  return P
end

# P function for a polytope
function PolytopeP(H, h, Γ)
  @assert true
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

#
function makeXk(k :: Int, b :: Int, Λ, η, ν, ffnet :: FeedForwardNetwork, pattern :: TPattern; seclow :: Float64 = 0.0, sechigh :: Float64 = 1.0)
  @assert ffnet.nettype isa ReluNetwork || ffnet.nettype isa TanhNetwork
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K

  Tdim = sum(ffnet.zdims[k+1:k+b])
  @assert size(Λ) == (Tdim, Tdim)
  @assert length(η) == length(ν) == Tdim

  T = makeT(Tdim, Λ, pattern)

  _Q11 = -2 * seclow * sechigh * T
  _Q12 = (seclow + sechigh) * T
  _Q13 = -(sechigh * ν + seclow * η)
  _Q22 = -2 * T
  _Q23 = ν + η
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])

  Ack = makeAck(k, b, ffnet)
  bck = makebck(k, b, ffnet)
  Bck = makeBck(k, b, ffnet)

  _R11 = Ack
  _R12 = bck
  _R21 = Bck
  _R22 = zeros(size(bck))
  _R31 = zeros(1, sum(ffnet.zdims[k:k+b]))
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  return R' * Q * R
end

#
#=
function makeXinit(b :: Int, γ, ffnet :: FeedForwardNetwork, input :: InputConstraint)
  return P
end
=#

# 
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
export e, E, Ec, F, FK, Faff
export makeA, makeb, makeB, makeAck, makebck, makeBck
export BoxP, PolytopeP
export makeXk, makeXinit, makeXsafe

end # End module

