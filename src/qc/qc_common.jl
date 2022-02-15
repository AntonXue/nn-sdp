# Make the A matrix
function makeA(nnet :: NeuralNetwork)
  edims = nnet.zdims[1:end-1]
  fdims = edims[2:end]
  Ws = [M[1:end, 1:end-1] for M in nnet.Ms]
  A = sum(E(k, fdims)' * Ws[k] * E(k, edims) for k in 1:(nnet.K-1))
  return A
end

# Make the bck stacked vector
function makeb(nnet :: NeuralNetwork)
  bs = [M[1:end, end] for M in nnet.Ms[1:end-1]]
  return vcat(bs...)
end

# Make the Bck matrix
function makeB(nnet :: NeuralNetwork)
  edims = nnet.zdims[1:end-1]
  fdims = edims[2:end]
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:(nnet.K-1))
  return B
end

