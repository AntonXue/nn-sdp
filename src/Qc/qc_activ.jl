using ..MyNeuralNetwork

include("qc_activ_bounded.jl")
include("qc_activ_sector.jl")

# Make the A matrix
function makeA(ffnet::FeedFwdNet)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  Ws = [M[1:end, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, fdims)' * Ws[k] * E(k, edims) for k in 1:(ffnet.K-1))
  return sparse(A)
end

# Make the b stacked vector
function makeb(ffnet::FeedFwdNet)
  bs = [M[1:end, end] for M in ffnet.Ms[1:end-1]]
  return vcat(bs...)
end

# Make the B matrix
function makeB(ffnet::FeedFwdNet)
  edims = ffnet.zdims[1:end-1]
  fdims = edims[2:end]
  B = sum(E(j, fdims)' * E(j+1, edims) for j in 1:(ffnet.K-1))
  return B
end

# Make the Zac; call this once for each Qac used
function makeZac(γac, qc::QcActiv, ffnet::FeedFwdNet)
  @assert length(γac) == qc.vardim
  Qac = makeQac(γac, qc)
  _R11 = makeA(ffnet)
  _R12 = makeb(ffnet)
  _R21 = makeB(ffnet)
  _R22 = zeros(size(_R21)[1])
  _R31 = zeros(1, size(_R21)[2])
  _R32 = 1
  R = [_R11 _R12; _R21 _R22; _R31 _R32]
  Zac = R' * Qac * R
  return Zac
end

# Do all the QCs that we could possibly use
function makeQcActivsIntvs(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, β::Int)
  @assert ffnet.activ isa ReluActiv || ffnet.activ isa TanhActiv
  @assert length(x1min) == length(x1max)

  # Interval propagation
  intv_info = Intervals.makeIntervalsInfo(x1min, x1max, ffnet)
  
  # Set up qc bounded
  acdim = sum(ffnet.xdims[2:end-1])
  acymin = vcat([acyi[1] for acyi in intv_info.x_intvs[2:end-1]]...)
  acymax = vcat([acyi[2] for acyi in intv_info.x_intvs[2:end-1]]...)
  qc_bounded = QcActivBounded(acydim=acdim, acymin=acymin, acymax=acymax)
  
  # Set up qc sector
  sec_acxmin = vcat([acxi[1] for acxi in intv_info.acx_intvs]...)
  sec_acxmax = vcat([acxi[2] for acxi in intv_info.acx_intvs]...)
  smin, smax = makeSectorMinMax(sec_acxmin, sec_acxmax, ffnet.activ)
  qc_sector = QcActivSector(activ=ffnet.activ, acxdim=acdim, β=β, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

  qc_activs = [qc_bounded; qc_sector]
  return qc_activs
end

# TODO: add more varieties as needed
function makeQcActivs(ffnet::FeedFwdNet; x1min = nothing, x1max = nothing, β=nothing)
  @assert x1min isa VecF64 && x1max isa VecF64 && β isa Int
  return makeQcActivsIntvs(ffnet, x1min, x1max, β)
end


