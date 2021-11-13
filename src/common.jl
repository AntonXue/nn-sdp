# Some functionalities that will be common across different algorithms
module Common

using ..Header

using LinearAlgebra

# The ith basis vector
function e(i :: Int, n :: Int)
  @assert 1 <= i <= n
  b = zeros(n)
  b[i] = 1.0
  return b
end

# The block index matrix for {zd[1], ..., zd[K], 1};
function E(k :: Int, zd :: Vector{Int})
  @assert 1 <= k <= length(zd)
  @assert zd[end] == 1
  width = sum(zd)
  low = sum(zd[1:k-1]) + 1
  high = sum(zd[1:k])
  B = zeros(zd[k], width)
  B[1:zd[k], low:high] = I(zd[k])
  return B
end

# The clique index matrix for {d[1], ..., d[K], d[K+1]}
function Ec(k :: Int, zd :: Vector{Int})
  lenzd = length(zd)
  @assert 1 <= k <= lenzd - 1
  @assert zd[end] == 1
  if k < lenzd - 1
    Ec = [E(k, zd); E(k+1, zd); E(lenzd, zd)]
  else
    Ec = [E(k, zd); E(1, zd); E(lenzd, zd)]
  end
  return Ec
end

# Ways to define Yk, for k < K the length of the network
function Yk(k, Q, ffnet :: FeedForwardNetwork)
  @assert k < ffnet.K
  xd = ffnet.xdims
  Wk = ffnet.M[k][1:end, 1:end-1]
  bk = ffnet.M[k][1:end, end]
  
  _R11 = Wk
  _R12 = zeros(xd[k+1], xd[k+1])
  _R13 = bk
  _R21 = zeros(xd[k+1], xd[k])
  _R22 = I(xd[k+1])
  _R23 = zeros(xd[k+1], 1)
  _R31 = zeros(1, xd[k])
  _R32 = zeros(1, xd[k+1])
  _R33 = 1
  R = [_R11 _R12 _R13; _R21 _R22 _R23; _R31 _R32 _R33]

  Y = R' * Q * R
  return Y
end

# YK, for the input and safety constraint
function YK(P, S, ffnet :: FeedForwardNetwork)
  (Ph, Pw) = size(P)
  (Sh, Sw) = size(S)

  _U11 = zeros(Sh-Ph, Sw-Pw)
  _U12 = zeros(Sh-Ph, Pw)
  _U22 = P
  U = [_U11 _U12; _U12' _U22]

  xd = ffnet.xdims
  K = ffnet.K
  WK = ffnet.M[K][1:end, 1:end-1]
  bK = ffnet.M[K][1:end, end]

  _R11 = WK
  _R12 = zeros(xd[K+1], xd[1])
  _R13 = bK
  _R21 = zeros(xd[1], xd[K])
  _R22 = I(xd[1])
  _R23 = zeros(xd[1], 1)
  _R31 = zeros(1, xd[K])
  _R32 = zeros(1, xd[1])
  _R33 = 1
  R = [_R11 _R12 _R13; _R21 _R22 _R23; _R31 _R32 _R33]

  Y = R' * U * R
  return Y
end

# Define the global QC for the ReLU function
function Qrelu(Λ, ν, η, α=0.0, β=1.0)
  @assert length(ν) == length(η)
  @assert size(Λ) == (length(ν), length(η))
  d = length(ν)
  T = zeros(d, d)
  for i = 1:d-1
    for j = i:d
      δij = e(i, d) - e(j, d)
      T = T + Λ[i,j] * δij * δij'
    end
  end

  _Q11 = -2 * α * β * (Diagonal(Λ) + T)
  _Q12 = (α + β) * (Diagonal(Λ) + T)
  _Q13 = -(β * ν + α * η)
  _Q22 = -2 * (Diagonal(Λ) + T)
  _Q23 = ν + η
  _Q33 = 0
  Q = [_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33]
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

export e, E, Ec
export Yk, YK, Qrelu, BoxP, PolytopeP

end # End module

