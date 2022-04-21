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

# Run interval propagation and quickly generate qc_activs
function makeQcActivs(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, β; use_qc_sector = true)
  # intv_info = Intervals.intervalsWorstCase(x1min, x1max, ffnet)
  # println("using worst case intervals")
  # intv_info = Intervals.intervalsRandomized(x1min, x1max, ffnet)
  # println("using sampled intervals")
  intv_info = Intervals.intervalsAutoLirpa(x1min, x1max, ffnet)
  println("using autolirpa intervals")
  
  # Set up qc bounded
  acdim = sum(ffnet.xdims[2:end-1])
  acymin = vcat([acyi[1] for acyi in intv_info.x_intvs[2:end-1]]...)
  acymax = vcat([acyi[2] for acyi in intv_info.x_intvs[2:end-1]]...)
  qc_bounded = QcActivBounded(acydim=acdim, acymin=acymin, acymax=acymax)
  
  if use_qc_sector
    sec_acxmin = vcat([acxi[1] for acxi in intv_info.acx_intvs]...)
    sec_acxmax = vcat([acxi[2] for acxi in intv_info.acx_intvs]...)
    smin, smax = findSectorMinMax(sec_acxmin, sec_acxmax, ffnet.activ)
    qc_sector = QcActivSector(activ=ReluActiv(), acxdim=acdim, β=β, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

    qc_activs = [qc_bounded; qc_sector]
  else
    qc_activs = [qc_bounded]
  end
  return qc_activs
end

