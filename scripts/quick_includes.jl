start_time = time()

using LinearAlgebra
using ArgParse
using Printf

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

nnet = loadNeuralNetwork(args["nnet"])

# Craft some artificial inputs
input = BoxInput(x1min=(ones(2) .- 0.1), x1max=(ones(2) .+ 0.1))

intv_info = intervalsWorstCase(input.x1min, input.x1max, nnet)

# QC bounded
qxdim = sum(nnet.xdims[2:end-1])
qxmin = vcat([qxi[1] for qxi in intv_info.x_intvs[2:end-1]]...)
qxmax = vcat([qxi[2] for qxi in intv_info.x_intvs[2:end-1]]...)
qcbounded_info = QcBoundedInfo(qxdim=qxdim, qxmin=qxmin, qxmax=qxmax)

# QC sector
qcsec_qxmin = vcat([prei[1] for prei in intv_info.qx_intvs]...)
qcsec_qxmax = vcat([prei[2] for prei in intv_info.qx_intvs]...)
smin, smax = sectorBounds(qcsec_qxmin, qcsec_qxmax, nnet.activ)
qcsector_info = QcSectorInfo(qxdim=qxdim, β=1, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

# qcinfos = [qcbounded_info]
# qcinfos = [qcsector_info]
qcinfos = [qcbounded_info, qcsector_info]

MOSEK_OPTS =
  Dict("MSK_IPAR_INTPNT_SOLVE_FORM" => 2)

deepsdp_opts = DeepSdpOptions(mosek_opts=MOSEK_OPTS, verbose=true)
# deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)

println("")

normal = [1.0; 1.0]
reach_soln = solveHplaneReach(nnet, input, qcinfos, deepsdp_opts, normal, verbose=true)
println(reach_soln)

println("\n\n")

safety_soln = solveSafetyL2Gain(nnet, input, qcinfos, deepsdp_opts, 1000.0, verbose=true)
println(safety_soln)

