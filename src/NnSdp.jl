module NnSdp

include("MyLinearAlgebra.jl");
include("MyNeuralNetwork.jl");
include("Qc/Qc.jl");
include("Intervals/Intervals.jl");
include("Methods/Methods.jl");
include("Utils/Utils.jl");

using Reexport
@reexport using .MyLinearAlgebra
@reexport using .MyNeuralNetwork
@reexport using .Qc
@reexport using .Intervals
@reexport using .Methods
@reexport using .Utils

# Default solver options
DEFAULT_MOSEK_OPTS = Dict()

# The generic formuation that everything ends up calling
function solveQuery(query::Query, opts::QueryOptions)
  soln = Methods.runQuery(query, opts)
  return soln
end

# Safety
function solveSafetyL2Gain(ffnet::FeedFwdNet, input::BoxInput, qcinfos, opts, L2gain::Float64; verbose = false)
  safety = L2gainSafety(L2gain, ffnet.xdims)
  safety_inst = SafetyQuery(ffnet=ffnet, input=input, output=safety, qcinfos=qcinfos)
  soln = Methods.runQuery(safety_inst, opts)
  return soln
end

# Load a P1
function solveHplaneReach(ffnet::FeedFwdNet, input::BoxInput, qcinfos, opts, normal::VecF64; verbose = false)
  hplane = HplaneReachSet(normal=normal)
  reach_inst = ReachQuery(ffnet=ffnet, input=input, reach=hplane, qcinfos=qcinfos)
  soln = Methods.runQuery(reach_inst, opts)
  return soln
end

# 
function runSafety(ffnet::FeedFwdNet, input::BoxInput, safety::SafetyConstraint, method;
                   mosek_opts = DEFAULT_MOSEK_OPTS,
                   β = 1,
                   verbose = false)
  println("Running safety verification with method $(method)")
  @assert method == :deepsdp || method == :chordalsdp

  # Run some interval propagations
  intv_info = intervalsWorstCase(input.x1min, input.x1max, ffnet)

  # QC bounded
  qxdim = sum(ffnet.xdims[2:end-1])
  qxmin = vcat([qxi[1] for qxi in intv_info.x_intvs[2:end-1]]...)
  qxmax = vcat([qxi[2] for qxi in intv_info.x_intvs[2:end-1]]...)
  qcbounded_info = QcBoundedInfo(qxdim=qxdim, qxmin=qxmin, qxmax=qxmax)

  # QC sector
  qcsec_qxmin = vcat([prei[1] for prei in intv_info.qx_intvs]...)
  qcsec_qxmax = vcat([prei[2] for prei in intv_info.qx_intvs]...)
  smin, smax = sectorBounds(qcsec_qxmin, qcsec_qxmax, ffnet.activ)
  qcsec_info = QcSectorInfo(qxdim=qxdim, β=β, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)
  
  qcinfos = [qcbounded_info, qcsec_info]

  if method == :deepsdp
    opts = DeepSdpOptions(use_dual=false, verbose=verbose)
  elseif method == :chordalsdp
    opts = ChordalSdpOptions(use_dual=false, verbose=verbose)
  else
    error("Unrecognized method: $(method)")
  end

  query = SafetyQuery(ffnet=ffnet, input=input, safety=safety, qcinfos=qcinfos)
  soln = Methods.runQuery(query, opts)
  return soln
end

export solveSafetyL2Gain, solveHplaneReach

end
