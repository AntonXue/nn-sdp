
function analyzeSparsity(qcinfos :: Vector{QcInfo})

end

# P function for a box
function makePbox(γin, x1min :: VecF64, x1max :: VecF64)
  @assert length(x1min) == length(x1max) == length(γin)
  Γ = Diagonal(γin)
  _P11 = -2 * Γ
  _P12 = Γ * (x1min + x1max)
  _P22 = -2 * x1min' * Γ * x1max
  P = [_P11 _P12; _P12' _P22]
  return P
end

# P function for a polytope
function makePpoly(Γ, H :: Matrix{Float64}, h :: VecF64)
  @assert size(H)[1] == length(h)
  _P11 = H' * Γ * H
  _P12 = -H' * Γ * h
  _P22 = h' * Γ * h
  P = [_P11 _P12; _P12' _P22]
  return P
end

# Bounding hyperplane such that normal' * f(x) <= h, for variable h
function makeShplane(normal :: VecF64, h, nnet :: NeuralNetwork)
  d1 = nnet.xdims[1]
  dK1 = nnet.xdims[end]
  @assert length(normal) == dK1
  _S11 = zeros(d1, d1)
  _S12 = zeros(d1, dK1)
  _S13 = zeros(d1)
  _S22 = zeros(dK1, dK1)
  _S23 = normal
  _S33 = -2 * h
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end

# γin is a vector
function makePin(γin, input :: InputConstraint, nnet :: NeuralNetwork)
  d1 = nnet.xdims[1]
  if input isa BoxInput
    @assert length(γin) == d1
    return makePbox(γin, input.x1min, input.x1max)
  elseif input isa PolyInput
    @assert length(γin) == d1^2
    Γ = reshape(γin, (d1, d1))
    return makePpolytope(input.H, input.h, Γ)
  else
    error(@sprintf("unsupported input constraints: %s", input))
  end
end

# Make the big Sout matrix, where S may be parametrized
function makeSout(S, nnet :: NeuralNetwork)
  WK = nnet.Ms[nnet.K][1:end, 1:end-1]
  bK = nnet.Ms[nnet.K][1:end, end]

  d1 = nnet.zdims[1]
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

