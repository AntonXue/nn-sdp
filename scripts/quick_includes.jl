start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using Dates

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "the NNet file location"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()
@printf("load done: %.3f\n", time() - start_time)

ffnet = loadFromNnet(args["nnet"])

# Craft some artificial inputs and safeties
qc_input = QcBoxInput(x1min=(ones(2) .- 0.1), x1max=(ones(2) .+ 0.1))
qc_safety = QcSafety(S=outNorm2S(1000, ffnet))
qc_reach = QcHplaneReach(normal=[1.0; 1.0])

intv_info = intervalsWorstCase(qc_input.x1min, qc_input.x1max, ffnet)

# QC bounded
acxdim = sum(ffnet.xdims[2:end-1])
acxmin = vcat([acxi[1] for acxi in intv_info.x_intvs[2:end-1]]...)
acxmax = vcat([acxi[2] for acxi in intv_info.x_intvs[2:end-1]]...)
qc_bnd = QcBoundedActiv(acxdim=acxdim, acxmin=acxmin, acxmax=acxmax)

# QC sector
qcsec_acxmin = vcat([acxi[1] for acxi in intv_info.acx_intvs]...)
qcsec_acxmax = vcat([acxi[2] for acxi in intv_info.acx_intvs]...)
smin, smax = findSectorMinMax(qcsec_acxmin, qcsec_acxmax, ffnet.activ)
qc_sec = QcSectorActiv(acxdim=acxdim, β=4, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

qc_activs = [qc_bnd, qc_sec]

safety_query = SafetyQuery(ffnet=ffnet, qc_input=qc_input, qc_safety=qc_safety, qc_activs=qc_activs)
reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_reach=qc_reach, qc_activs=qc_activs)


mosek_opts = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true)
deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)

chordal_reach_soln = runQuery(reach_query, chordalsdp_opts)
deep_reach_soln = runQuery(reach_query, deepsdp_opts)

chordal_safety_soln = runQuery(safety_query, chordalsdp_opts)
deep_safety_soln = runQuery(safety_query, deepsdp_opts)


println("")


