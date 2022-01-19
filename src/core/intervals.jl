# Functionalties for interval propagation

module Intervals

using ..Header
using ..Common
using Printf

# Calculate slope bounds for the input interval [ymin, ymax]
# Calculate slopes for when 
function slopeBounds(ymin :: Vector{Float64}, ymax :: Vector{Float64}, ffnet :: FeedForwardNetwork)
  @assert length(ymin) == length(ymax)
  ε = 1e-4
  if ffnet.type isa ReluNetwork
    Ipos = findall(z -> z > ε, ymin)
    Ineg = findall(z -> z < -ε, ymax)
    ϕa = zeros(length(ymin))
    ϕa[Ipos] .= 1.0
    ϕb = ones(length(ymax))
    ϕb[Ineg] .= 0.0
    return ϕa, ϕb

  elseif ffnet.type isa TanhNetwork
    # Default values
    ϕa, ϕb = zeros(length(ymin)), ones(length(ymax))
    for i in 1:length(ymin)
      if ymin[i] * ymax[i] >= 0
        ϕa[i] = tanh(ymax[i]) / ymax[i]
        ϕb[i] = tanh(ymin[i]) / ymin[i]
      else
        ϕa[i] = min(tanh(ymin[i]) / ymin[i], tanh(ymax[i]) / ymax[i])
        ϕb[i] = 1
      end
    end
    return ϕa, ϕb
  else
    error(@sprintf("unsupported network: %s", ffnet))
  end
end

# Given k, b, ϕin_intvs, find the y[k], ..., x[k+b-1]
# We should have length(ϕin_intvs) == K - 1
function _selectϕinIntervals(k :: Int, b :: Int, ϕin_intvs :: Vector{Tuple{Vector{Float64}, Vector{Float64}}})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(ϕin_intvs) + 1 # K
  ymin = vcat([yi[1] for yi in ϕin_intvs[k:k+b-1]]...)
  ymax = vcat([yi[2] for yi in ϕin_intvs[k:k+b-1]]...)
  return ymin, ymax
end

function selectϕinIntervals(k :: Int, b :: Int, ϕin_intvs)
  if ϕin_intvs isa Nothing
    return nothing
  else
    return _selectϕinIntervals(k, b, ϕin_intvs)
  end
end

# Given k, b, x_intvs, find the x[k+1], ..., x[k+b]
# We should have length(x_intvs) == length(zdims) == K + 1
function _selectϕoutIntervals(k :: Int, b :: Int, x_intvs :: Vector{Tuple{Vector{Float64}, Vector{Float64}}})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(x_intvs) - 1 # K
  ϕmin = vcat([xi[1] for xi in x_intvs[k+1:k+b]]...)
  ϕmax = vcat([xi[2] for xi in x_intvs[k+1:k+b]]...)
  return ϕmin, ϕmax
end

function selectϕoutIntervals(k :: Int, b :: Int, x_intvs)
  if x_intvs isa Nothing
    return nothing
  else
    return _selectϕoutIntervals(k, b, x_intvs)
  end
end

# Given k, b, slope_intvs, find the ones at ϕ[k], ..., ϕ[k+b-1]
# We should have length(slope_intvs) == K - 1
function _selectSlopeIntervals(k :: Int, b :: Int, slope_intvs :: Vector{Tuple{Vector{Float64}, Vector{Float64}}})
  @assert k >= 1 && b >= 1
  @assert 1 <= k + b <= length(slope_intvs) + 1 # K
  ϕa = vcat([si[1] for si in slope_intvs[k:k+b-1]]...)
  ϕb = vcat([si[2] for si in slope_intvs[k:k+b-1]]...)
  return ϕa, ϕb
end

function selectSlopeIntervals(k :: Int, b :: Int, slope_intvs)
  if slope_intvs isa Nothing
    return nothing
  else
    return _selectSlopeIntervals(k, b, slope_intvs)
  end
end

# Randomized propagation
function randomizedPropagation(x1min :: Vector{Float64}, x1max :: Vector{Float64}, ffnet :: FeedForwardNetwork, N :: Int)
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]

  # The activation function
  function ϕ(x)
    if ffnet.type isa ReluNetwork; return max.(x, 0)
    elseif ffnet.type isa TanhNetwork; return tanh.(x)
    else; error(@sprintf("unsupported network: %s", ffnet))
    end
  end

  xgaps = x1max - x1min
  points = rand(ffnet.xdims[1], N)
  X1s = hcat([x1min + (p .* xgaps) for p in eachcol(points)]...)

  # Each x[1], x[2], ..., x[K], x[K+1] of the network
  x_intvs = Vector{Any}()
  push!(x_intvs, (x1min, x1max))

  # The inputs to the activation functions; should be K-1 of them
  ϕin_intvs = Vector{Any}()

  # All the slope bounds; should be K-1 of them
  slope_intvs = Vector{Any}()

  Xks = X1s
  for (k, Mk) in enumerate(ffnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    Yks = hcat([Wk * xk + bk for xk in eachcol(Xks)]...)
    ykmin, ykmax = [minimum(row) for row in eachrow(Yks)], [maximum(row) for row in eachrow(Yks)]

    if k == ffnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(ϕin_intvs, (ykmin, ykmax))
      ϕa, ϕb = slopeBounds(ykmin, ykmax, ffnet)
      push!(slope_intvs, (ϕa, ϕb))
      Xks = hcat([ϕ(yk) for yk in eachcol(Yks)]...)
      xkmin, xkmax = [minimum(row) for row in eachrow(Xks)], [maximum(row) for row in eachrow(Xks)]
    end
    push!(x_intvs, (xkmin, xkmax))
  end

  # Each is a list of tuple of vectors, for flexibility; vcat as needed
  return x_intvs, ϕin_intvs, slope_intvs
end

# Worst case propagation of a box
function worstCasePropagation(x1min :: Vector{Float64}, x1max :: Vector{Float64}, ffnet :: FeedForwardNetwork)
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]

  # The activation function
  function ϕ(x)
    if ffnet.type isa ReluNetwork; return max.(x, 0)
    elseif ffnet.type isa TanhNetwork; return tanh.(x)
    else; error(@sprintf("unsupported network: %s", ffnet))
    end
  end

  # Each x[1], x[2], ..., x[K], x[K+1] of the network
  x_intvs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
  push!(x_intvs, (x1min, x1max))

  # The inputs to the activation functions; should be K-1 of them
  ϕin_intvs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()

  # All the slope bounds; should be K-1 of them
  slope_intvs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()

  xkmin, xkmax = x1min, x1max
  for (k, Mk) in enumerate(ffnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk

    if k == ffnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(ϕin_intvs, (ykmin, ykmax))
      ϕa, ϕb = slopeBounds(ykmin, ykmax, ffnet)
      push!(slope_intvs, (ϕa, ϕb))
      xkmin, xkmax = ϕ(ykmin), ϕ(ykmax)
    end
    push!(x_intvs, (xkmin, xkmax))
  end

  # Each is a list of tuple of vectors, for flexibility; vcat as needed
  return x_intvs, ϕin_intvs, slope_intvs
end

export selectϕinIntervals, selectϕoutIntervals, selectSlopeIntervals
export randomizedPropagation, worstCasePropagation

end # End module

