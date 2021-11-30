# Some functionalities that will be common across different algorithms
module Common

using ..Header
using LinearAlgebra
using JuMP

# The ith basis vector
function e(i :: Int, n :: Int)
  @assert 1 <= i <= n
  e = zeros(n)
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

# The clique index of stride s that selects {x[k], ..., x[k+s], x[aff]}
function Ec(k :: Int, zdims :: Vector{Int}; stride :: Int = 1)
  @assert 1 <= k && 1 <= stride
  @assert 1 <= k + stride + 1 <= length(zdims)
  @assert zdims[end] == 1 # Affine component
  Ecks = [E(j, zdims) for j in k:k+stride]
  return [vcat(Ecks...); E(length(zdims), zdims)]
end

# Selects specific elements within the clique {x[k], ..., x[k+s], x[aff]}
function Eck(k :: Int, i :: Int, zdims :: Vector{Int}; stride :: Int = 1)
  @assert 1 <= k && 1 <= stride
  @assert 1 <= k + stride + 1 <= length(zdims)
  @assert 1 <= i <= stride + 2
  @assert zdims[end] == 1
  return E(i, [zdims[k:k+stride]; 1])
end

# Make the matrices into block diagonal form
function blockdiag(Xs :: Vector{Matrix{T}}) where {T <: Any}
  lenXs = length(Xs)
  @assert lenXs > 0
  Xhs = [size(X)[1] for X in Xs]
  Xws = [size(X)[2] for X in Xs]
  height = sum(Xhs)
  Bs = zeros(height, 0)
  for k in 1:lenXs
    Bstart = sum(Xhs[1:k-1]) + 1
    Bend = sum(Xhs[1:k-1]) + Xhs[k]
    Bk = zeros(height, Xws[k])
    Bk[Bstart:Bend, 1:end] = Xs[k]
    Bs = [Bs Bk]
  end
  return Bs
end

# Make the A, B blocks that will get multiplied by the Q
function Qsides(k :: Int, ffnet :: FeedForwardNetwork; stride :: Int = 1)
  @assert 1 <= k && 1 <= stride
  @assert 1 <= k + stride + 1 <= length(ffnet.zdims)

  height = sum(ffnet.xdims[(k+1):(k+stride)])

  _A1 = blockdiag([ffnet.Ms[j][1:end, 1:end-1] for j in k:(k+stride-1)])
  _A2 = zeros(height, ffnet.xdims[k+stride])
  A = [_A1 _A2]

  b = vcat([ffnet.Ms[j][1:end, end] for j in k:(k+stride-1)]...)

  _B1 = zeros(height, ffnet.xdims[k])
  _B2 = blockdiag([Matrix(I(ffnet.xdims[j])) for j in (k+1):(k+stride)])
  B = [_B1 _B2]
  return (A, b, B)
end

# The kth Y given a stride
function Y(k :: Int, Q, ffnet :: FeedForwardNetwork; stride :: Int = 1)
  @assert 1 <= k && 1 <= stride
  @assert 1 <= k + stride + 1 <= length(ffnet.zdims)

  qxdim = sum(ffnet.zdims[j] for j in (k+1):(k+stride))
  qdim = 2 * qxdim + 1
  @assert size(Q) == (qdim, qdim)

  (A, b, B) = Qsides(k, ffnet, stride=stride)

  _S11 = A
  _S12 = b
  _S21 = B
  _S22 = zeros(size(b))
  _S31 = zeros(1, sum(ffnet.xdims[k:k+stride]))
  _S32 = 1
  S = [_S11 _S12; _S21 _S22; _S31 _S32]
  return S' * Q * S
end

# The Y for the input constraint
function Yinput(P, ffnet :: FeedForwardNetwork; stride :: Int = 1)
  @assert size(P) == (ffnet.xdims[1] + 1, ffnet.xdims[1] + 1)
  _Ecinput1 = Eck(1, 1, ffnet.zdims, stride=stride)
  _Ecinput2 = Eck(1, 1 + stride + 1, ffnet.zdims, stride=stride)
  Ecinput = [_Ecinput1; _Ecinput2]
  return Ecinput' * P * Ecinput
end

# The Y for the safety constraint
function Ysafety(S, ffnet :: FeedForwardNetwork; stride :: Int = 1)
  @assert size(S) == (ffnet.xdims[end] + 1, ffnet.xdims[end] + 1)

  _S11 = S[1:end-1, 1:end-1]
  _S12 = S[1:end-1, end]
  _S22 = S[end, end]

  WK = ffnet.Ms[end][1:end, 1:end-1]
  bK = ffnet.Ms[end][1:end, end]

  _T11 = WK' * _S11 * WK
  _T12 = WK' * _S11 * bK + WK' * _S12
  _T22 = bK' * _S11 * bK + 2 * (bK' * _S12) + _S22
  T = [_T11 _T12; _T12' _T22]

  p = ffnet.K - stride
  _Ecsafety1 = Eck(p, 1 + stride, ffnet.zdims, stride=stride)
  _Ecsafety2 = Eck(p, 1 + stride + 1, ffnet.zdims, stride=stride)
  Ecsafety = [_Ecsafety1; _Ecsafety2]
  return Ecsafety' * T * Ecsafety
end

function Tλ(dim :: Int64, Λ)
  T_start_time = time()
  println("calling T")

  @assert size(Λ) == (dim, dim)

  Δ = vcat([e(i, dim)' - e(j, dim)' for i in 1:(dim-1) for j in (i+1):dim]...)
  U = diagm(vec([Λ[i,j] for i in 1:(dim-1) for j in (i+1):dim]))

  println("Δ size: " * string(size(Δ)))
  println("U size: " * string(size(U)))

  T = Δ' * U * Δ

  println("returning from T, time: " * string(time() - T_start_time))
  return T
end

# Define the global QC for the ReLU function
function Qrelu(qxdim :: Int64, Λ, η, ν; α :: Float64 = 0.0, β :: Float64 = 1.0)
  qrelu_start_time = time()
  @assert qxdim == length(ν) == length(η)
  @assert size(Λ) == (qxdim, qxdim)

  T = Tλ(qxdim, Λ)

  println("qrelu: finished setting up T: " * string(time() - qrelu_start_time))

  V = diagm(diag(Λ)) + T

  _Q11 = -2 * α * β * V
  _Q12 = (α + β) * V
  _Q13 = -(β * ν + α * η)
  _Q22 = -2 * V
  _Q23 = ν + η
  _Q33 = 0

  Q = [_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33]
  println("qrelu: returning, " * string(time() - qrelu_start_time))
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

# Some variants that are parameterized with a single γ vector

# Slowly using up the English alphabet
export e, E, Ec, Eck
export Y, Yinput, Ysafety
export Tλ, Qsides, Qrelu
export BoxP, PolytopeP

end # End module

