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
  T = diagm(zeros(Tdim))
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
  elseif pattern isa FullyDensePattern
    @assert size(Λ) == (Tdim, Tdim)
    return makeBandedT(Tdim, Λ, Tdim)
  else
    error("unsupported pattern: " * string(pattern))
  end
end

# P function for a box
function makeBoxP(xbot, xtop, γ)
  @assert length(xbot) == length(xtop) == length(γ)
  Γ = Diagonal(γ)
  _P11 = -2 * Γ
  _P12 = Γ * (xbot + xtop)
  _P22 = -2 * xbot' * Γ * xtop
  P = [_P11 _P12; _P12' _P22]
  return P
end

# P function for a polytope
function makePolytopeP(H, h, Γ)
  @assert true
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

# Bounding hyperplane such that c' * f(x) <= d, for variable d
function makeHyperplaneS(c :: Vector{Float64}, d, ffnet :: FeedForwardNetwork)
  d1 = ffnet.xdims[1]
  dK1 = ffnet.xdims[end]
  @assert length(c) == dK1
  _S11 = zeros(d1, d1)
  _S12 = zeros(d1, dK1)
  _S13 = zeros(d1)
  _S22 = zeros(dK1, dK1)
  _S23 = c
  _S33 = -d
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end

# The Q matirx
function makeQrelu(Tdim :: Int, Λ, η, ν, pattern :: TPattern; sbots = zeros(Tdim), stops = ones(Tdim))
  @assert size(Λ) == (Tdim, Tdim)
  @assert length(η) == length(ν) == Tdim

  T = makeT(Tdim, Λ, pattern)

  sbot0, stop0 = 0, 1

  λ = diag(Λ)
  _Q11 = -2 * diagm(sbots .* stops .* λ) - 2 * (sbot0 * stop0 * T)
  _Q12 = diagm((sbots + stops) .* λ) + (sbot0 + stop0) * T
  _Q13 = -(stop0 .* ν) - (sbot0 .* η)
  _Q22 = -2 * T
  _Q23 = ν + η
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

#
function makeXk(k :: Int, b :: Int, Λ, η, ν, ffnet :: FeedForwardNetwork, pattern :: TPattern; sbots = zeros(length(η)), stops = ones(length(η)))
  @assert ffnet.nettype isa ReluNetwork
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K

  Tdim = sum(ffnet.zdims[k+1:k+b])
  @assert size(Λ) == (Tdim, Tdim)
  @assert length(η) == length(ν) == Tdim

  if ffnet.nettype isa ReluNetwork
    Q = makeQrelu(Tdim, Λ, η, ν, pattern, sbots=sbots, stops=stops)
  else
    error("unsupported network type: " * string(ffnet.nettype))
  end

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

# γ is a vector
function makeXinit(γ, input :: InputConstraint, ffnet :: FeedForwardNetwork)
  d1 = ffnet.xdims[1]
  if input isa BoxConstraint
    @assert length(γ) == d1
    return makeBoxP(input.xbot, input.xtop, γ)
  elseif input isa PolytopeConstraint
    @assert length(γ) == d1^2
    Γ = reshape(γ, (d1, d1))
    return makePolytopeP(input.H, input.h, Γ)
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

# Count overlaps
function makeΩ(b :: Int, zdims :: Vector{Int})
  @assert 1 <= length(zdims) - b - 2
  @assert zdims[end] == 1
  p = length(zdims) - b - 2
  Ω = zeros(sum(zdims), sum(zdims))
  for k in 1:p
    Eck = Ec(k, b, zdims)
    height = size(Eck)[1]
    Ω = Ω + Eck' * (fill(1, (height, height))) * Eck
  end
  return Ω
end

# The ℧ matrix that is the "inverse" scaling of Ω
function makeΩinv(b :: Int, zdims :: Vector{Int})
  @assert 1 <= length(zdims) - b - 2
  @assert zdims[end] == 1

  Ω = makeΩ(b, zdims)
  Ωinv = 1 ./ Ω
  Ωinv[isinf.(Ωinv)] .= 0
  return Ωinv
end

# Calculate the slope upper / lower bounds of each layer
function propagateBox(xbot :: Vector{Float64}, xtop :: Vector{Float64}, ffnet :: FeedForwardNetwork)
  function ϕ(x)
    if ffnet.nettype isa ReluNetwork; return max.(x, 0)
    elseif ffnet.nettype isa TanhNetwork; return tanh.(x)
    else; error("unsupported network: " * string(ffnet))
    end
  end

  @assert length(xbot) == length(xtop) == ffnet.xdims[1]


  # The inputs to the activation functions; should be K-1 of them
  slims = Vector{Any}()

  # The state limits right after each activation function
  xlims = Vector{Any}()
  push!(xlims, (xbot, xtop))
  xkbot, xktop = xbot, xtop

  # Run through each layer
  for (k, Mk) in enumerate(ffnet.Ms)
    # Generate the vertices of the hypercube for each iteration
    xkverts = vec(collect(Iterators.product(zip(xkbot, xktop)...)))
    xkverts = [[i for i in v] for v in xkverts]

    ykverts = hcat([Mk * [xkv; 1] for xkv in xkverts]...)
    ykbot, yktop = minimum(ykverts, dims=2), maximum(ykverts, dims=2)

    if ffnet.nettype isa ReluNetwork
      skbot = [if ykb >= 0 && ykt >= 0; 1 else 0 end for (ykb, ykt) in zip(ykbot, yktop)]
      sktop = [if ykb <= 0 && ykt <= 0; 0 else 1 end for (ykb, ykt) in zip(ykbot, yktop)]
    else
      error("unsupported network: " * string(ffnet))
    end

    if k == ffnet.K
      zkverts = ykverts
    else
      zkverts = hcat([ϕ(yk) for yk in eachcol(ykverts)]...)

      # Only push if we are in an intermediate layer
      push!(slims, (skbot, sktop))
    end

    xkbot, xktop = minimum(zkverts, dims=2), maximum(zkverts, dims=2)

    # Push
    push!(xlims, (xkbot, xktop))
  end

  return xlims, slims
end

# Slowly using up the English alphabet
export e, E, Ec, F, FK, Faff
export makeA, makeb, makeB, makeAck, makebck, makeBck
export makeQrelu
export makeBoxP, makePolytopeP
export makeHyperplaneS
export makeXk, makeXinit, makeXsafe
export makeΩ, makeΩinv
export propagateBox

end # End module

