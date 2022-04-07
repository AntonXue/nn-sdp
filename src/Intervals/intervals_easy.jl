# Worst case propagation of a box
function intervalsWorstCase(x1min::VecF64, x1max::VecF64, ffnet::FeedFwdNet)
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]

  # The activation function
  function ϕ(x)
    if ffnet.activ isa ReluActiv; return max.(x, 0)
    elseif ffnet.activ isa TanhActiv; return tanh.(x)
    else; error("unsupported network: $(ffnet)")
    end
  end

  # Each x[1], x[2], ..., x[K], x[K+1] of the network
  x_intvs = Vector{PairVecF64}()
  push!(x_intvs, (x1min, x1max))

  # The inputs to ϕ[1], ..., ϕ[K-1]
  acx_intvs = Vector{PairVecF64}()

  xkmin, xkmax = x1min, x1max
  for (k, Mk) in enumerate(ffnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk

    if k == ffnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(acx_intvs, (ykmin, ykmax))
      xkmin, xkmax = ϕ(ykmin), ϕ(ykmax)
    end
    push!(x_intvs, (xkmin, xkmax))
  end

  # Each is a list of tuple of vectors for flexibility; vcat as needed
  return IntervalInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs)
end

# Randomized propagation. Not sound
function intervalsRandomized(x1min::VecF64, x1max::VecF64, ffnet::FeedFwdNet; N = Int(1e6))
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]

  # The activation function
  function ϕ(x)
    if ffnet.activ isa ReluActiv; return max.(x, 0)
    elseif ffnet.activ isa TanhActiv; return tanh.(x)
    else; error("unsupported network: $(ffnet)")
    end
  end

  # The gap
  xgaps = x1max - x1min
  points = rand(ffnet.xdims[1], N)
  X1s = hcat([x1min + (p .* xgaps) for p in eachcol(points)]...)

  # Each x[1], x[2], ..., x[K], x[K+1] of the network
  x_intvs = Vector{PairVecF64}()
  push!(x_intvs, (x1min, x1max))

  # The inputs to ϕ[1], ..., ϕ[K-1]
  acx_intvs = Vector{PairVecF64}()

  Xks = X1s
  for (k, Mk) in enumerate(ffnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    Yks = hcat([Wk * xk + bk for xk in eachcol(Xks)]...)
    ykmin, ykmax = [minimum(row) for row in eachrow(Yks)], [maximum(row) for row in eachrow(Yks)]

    if k == ffnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(acx_intvs, (ykmin, ykmax))
      xkmin, xkmax = ϕ(ykmin), ϕ(ykmax)
      Xks = hcat([ϕ(yk) for yk in eachcol(Yks)]...)
    end
    push!(x_intvs, (xkmin, xkmax))
  end
  # Each is a list of tuple of vectors for flexibility; vcat as needed
  return IntervalInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs)
end

