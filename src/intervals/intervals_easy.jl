# Worst case propagation of a box
function intervalsWorstCase(x1min :: VecF64, x1max :: VecF64, nnet :: NeuralNetwork)
  @assert length(x1min) == length(x1max) == nnet.xdims[1]

  # The activation function
  function ϕ(x)
    if nnet.activ isa ReluActivation; return max.(x, 0)
    elseif nnet.activ isa TanhActivation; return tanh.(x)
    else; error(@sprintf("unsupported network: %s", nnet))
    end
  end

  # Each x[1], x[2], ..., x[K], x[K+1] of the network
  x_intvs = Vector{PairVecF64}()
  push!(x_intvs, (x1min, x1max))

  # The inputs to ϕ[1], ..., ϕ[K-1]
  ac_intvs = Vector{PairVecF64}()

  xkmin, xkmax = x1min, x1max
  for (k, Mk) in enumerate(nnet.Ms)
    Wk, bk = Mk[1:end, 1:end-1], Mk[1:end, end]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk

    if k == nnet.K
      xkmin, xkmax = ykmin, ykmax
    else
      push!(ac_intvs, (ykmin, ykmax))
      xkmin, xkmax = ϕ(ykmin), ϕ(ykmax)
    end
    push!(x_intvs, (xkmin, xkmax))
  end

  # Each is a list of tuple of vectors, for flexibility; vcat as needed
  return IntervalInfo(nnet=nnet, x_intvs=x_intvs, ac_intvs=ac_intvs)
end

