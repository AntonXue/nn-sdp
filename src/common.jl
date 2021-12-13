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
function makePpolytope(H , h, Γ)
  @assert true
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

# Bounding hyperplane such that c' * f(x) <= d, for variable d
function makeShyperplane(c :: Vector{Float64}, d, ffnet :: FeedForwardNetwork)
  d1 = ffnet.xdims[1]
  dK1 = ffnet.xdims[end]
  @assert length(c) == dK1
  _S11 = zeros(d1, d1)
  _S12 = zeros(d1, dK1)
  _S13 = zeros(d1)
  _S22 = zeros(dK1, dK1)
  _S23 = c
  _S33 = -2 * d
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end

# Make a T matrix of a particular size given matrix of variables τ
function makeT(dim :: Int, τ)
  @assert dim >= 1
  @assert size(τ) == (dim, dim)
  ijs = [(i, j) for i in 1:(dim-1) for j in (i+1):dim]
  δts = [e(i, dim)' - e(j, dim)' for (i, j) in ijs]
  Δ = vcat(δts...)
  V = diagm(vec([τ[i,j] for (i, j) in ijs]))
  T = Δ' * V * Δ
  return T
end

# Make Q with help of xlims where we are known to have bounded activation
function makeQbounded(qxdim :: Int, d, xlims :: Tuple{Vector{Float64}, Vector{Float64}})
  xmin, xmax = xlims
  @assert all(xmin .<= xmax)
  @assert -Inf < minimum(xmin) && maximum(xmax) < Inf
  @assert length(d) == length(xmin) == length(xmax) == qxdim

  D = diagm(d)
  _Q11 = zeros(qxdim, qxdim)
  _Q12 = zeros(qxdim, qxdim)
  _Q13 = zeros(qxdim)
  _Q22 = -2 * D
  _Q23 = D * (xmin + xmax)
  _Q33 = -2 * xmin' * D * xmax
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

# Make the Q relu, optionally enabling localized slope limits
function makeQrelu(qxdim :: Int, λ, τ, η, ν; slims = (zeros(qxdim), ones(qxdim)))
  @assert length(λ) == length(η) == length(ν) == qxdim
  @assert size(τ) == (qxdim, qxdim)

  ε = 1e-6
  smin, smax = slims
  @assert -ε <= minimum(smin) && maximum(smin) <= 1 + ε
  @assert -ε <= minimum(smax) && maximum(smax) <= 1 + ε

  T = makeT(qxdim, τ)
  s0, s1 = 0, 1
  _Q11 = -2 * diagm(smin .* smax .* λ) - 2 * (s0 * s1 * T)
  _Q12 = diagm((smin + smax) .* λ) + (s0 + s1) * T
  _Q13 = -(smin .* η) - (smax .* ν)
  _Q22 = -2 * T
  _Q23 = η + ν
  _Q33 = 0
  Q = Symmetric([_Q11 _Q12 _Q13; _Q12' _Q22 _Q23; _Q13' _Q23' _Q33])
  return Q
end

# Make an Xk, optionally enabling x bounds and slope limits
function makeXk(k :: Int, b :: Int, vars, ffnet :: FeedForwardNetwork; xlims = nothing, slims = nothing)
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= ffnet.K
  qxdim = sum(ffnet.zdims[k+1:k+b])

  # For the relu network
  if ffnet.type isa ReluNetwork
    λ, τ, η, ν, d = vars
    @assert length(λ) == length(η) == length(ν) == length(d) == qxdim
    @assert size(τ) == (qxdim, qxdim)

    # If slope limits are non-trivial, use them
    if slims isa Nothing
      Q = makeQrelu(qxdim, λ, τ, η, ν)
    else
      Q = makeQrelu(qxdim, λ, τ, η, ν, slims=slims)
    end

    # If x limits exist and are not infinite, use them
    if !(xlims isa Nothing) && (-Inf < minimum(xlims[1]) && maximum(xlims[2]) < Inf)
      Q = Q + makeQbounded(qxdim, d, xlims)
    end

  # Other kinds
  else
    error("unsupported network: " * string(ffnet.type))
  end

  _R11 = makeAck(k, b, ffnet)
  _R12 = makebck(k, b, ffnet)
  _R21 = makeBck(k, b, ffnet)
  _R22 = zeros(size(_R12))
  _R31 = zeros(1, size(_R21)[2])
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  return R' * Q * R
end

# γ is a vector
function makeXinit(γ, input :: InputConstraint, ffnet :: FeedForwardNetwork)
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

# Propagate a box through the network
function propagateBox(x1min :: Vector{Float64}, x1max :: Vector{Float64}, ffnet :: FeedForwardNetwork)
  function ϕ(x)
    if ffnet.type isa ReluNetwork; return max.(x, 0)
    elseif ffnet.type isa TanhNetwork; return tanh.(x)
    else; error("unsupported network: " * string(ffnet))
    end
  end

  @assert length(x1min) == length(x1max) == ffnet.xdims[1]

  # The inputs to the activation functions; should be K-1 of them
  ylimss = Vector{Any}()

  # The state limits right after each activation
  xlimss = Vector{Any}()
  push!(xlimss, (x1min, x1max))
  xkmin, xkmax = x1min, x1max

  for (k, Mk) in enumerate(ffnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk

    if k == ffnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(ylimss, (ykmin, ykmax))
      xkmin, xkmax = ϕ(ykmin), ϕ(ykmax)
    end
    push!(xlimss, (xkmin, xkmax))
  end
  return xlimss, ylimss
end


# Slowly using up the English alphabet
export e, E, Ec, F, FK, Faff
export makeA, makeb, makeB, makeAck, makebck, makeBck
export makeQbounded, makeQrelu
export makePbox, makePpolytope
export makeShyperplane
export makeXk, makeXinit, makeXsafe
export makeΩ, makeΩinv
export propagateBox

end # End module

