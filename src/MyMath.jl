module MyMath

using LinearAlgebra
using SparseArrays

# Custom type definitions
const VecInt = Vector{Int}
const VecReal = Vector{<:Real}
const PairVecReal = Tuple{<:VecReal, <:VecReal}
const MatReal = AbstractMatrix{<:Real}
const SymReal = Symmetric{<:Real}

# Splice a vector
function splice(x, sizes::VecInt)
  @assert all(sizes .>= 0)
  @assert 1 <= length(x) == sum(sizes)
  num_sizes = length(sizes)
  highs = [sum(sizes[1:k]) for k in 1:num_sizes]
  lows = [1; [1 + highk for highk in highs[1:end-1]]]
  @assert length(highs) == length(lows)
  splices = [x[lows[k] : highs[k]] for k in 1:num_sizes]
  return splices
end

# The ith basis vector
function e(i::Int, dim::Int)
  @assert 1 <= i <= dim
  e = spzeros(dim)
  e[i] = 1
  return e
end

# The ith block index matrix
function E(i::Int, dims::VecInt)
  @assert 1 <= i <= length(dims)
  width = sum(dims)
  low = sum(dims[1:i-1]) + 1
  high = sum(dims[1:i])
  E = spzeros(dims[i], width)
  E[1:dims[i], low:high] = I(dims[i])
  return E
end

# Make a clique based on hot elements
function Ec(elems::VecInt, N::Int)
  @assert elems == unique(elems) == sort(elems)
  @assert length(elems) >= 1
  @assert 1 <= elems[1] && elems[end] <= N
  eis = [e(i, N)' for i in elems]
  return vcat(eis...)
end

export VecInt, VecReal, PairVecReal, MatReal, SymReal
export splice, e, E, Ec

end

