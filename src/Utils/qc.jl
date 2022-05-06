# Various utilities for constructing QCs that don't cleanly fit into src/Qc/
using LinearAlgebra

# A general form of quadratic safety: a||x||^2 + b||f(x)||^2 + c <= 0
function abcQuadS(a, b, c, ffnet::FeedFwdNet)
  xdims, K = ffnet.xdims, ffnet.K
  _S11 = a * I(xdims[1])
  _S12 = spzeros(xdims[1], xdims[K+1])
  _S13 = spzeros(xdims[1], 1)
  _S22 = b * I(xdims[K+1])
  _S23 = spzeros(xdims[K+1], 1)
  _S33 = c
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
end

# ||f(x)||^2 <= L ||x||^2
function L2S(L2gain, ffnet::FeedFwdNet)
  return abcQuadS(-L2gain, 1.0, 0.0, ffnet)
end

# ||f(x)||^2 <= C
function outNorm2S(norm2, ffnet::FeedFwdNet)
  return abcQuadS(0.0, 1.0, -norm2, ffnet)
end

# Hyperplane S
function hplaneS(normal, h, ffnet::FeedFwdNet)
  xdims, K = ffnet.xdims, ffnet.K
  _S11 = spzeros(xdims[1], xdims[1])
  _S12 = spzeros(xdims[1], xdims[K+1])
  _S13 = spzeros(xdims[1])
  _S22 = spzeros(xdims[K+1], xdims[K+1])
  _S23 = normal
  _S33 = -2 * h
  S = [_S11 _S12 _S13; _S12' _S22 _S23; _S13' _S23' _S33]
  return S
end

# Uniformly sample a bunch of trajectories from an interval
function sampleTrajs(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, N::Int=100000)
  @assert length(x1min) == length(x1max) == ffnet.xdims[1]
  xgaps = x1max - x1min
  box01points = rand(ffnet.xdims[1], N)
  x1s = [x1min + (p .* xgaps) for p in eachcol(box01points)]
  ys = [evalFeedFwdNet(ffnet, x1) for x1 in x1s]
  return ys
end

# Gives the covariance matrix Σ and y0 such that
function approxEllipsoid(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, N::Int=100000)
  ys = sampleTrajs(ffnet, x1min, x1max, N)
  y0 = sum(ys) / length(ys)
  Y = hcat(ys...)
  Y0 = y0 * ones(1, length(ys))
  P = (Y - Y0) * (Y - Y0)'

  λs = eigvals(P)
  μ = (λs[1] + λs[end]) / 2
  P = P + μ* I
  return P, y0
end

