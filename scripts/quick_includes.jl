start_time = time()

using LinearAlgebra
using ArgParse
using Printf

include("../src/nn_sdp.jl"); using .NnSdp
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
acmin = vcat([xi[1] for xi in intv_info.x_intvs[2:end-1]]...)
acmax = vcat([xi[2] for xi in intv_info.x_intvs[2:end-1]]...)
qcbounded_info = QcBoundedInfo(qxdim=qxdim, acmin=acmin, acmax=acmax)

# QC sector
pre_a = vcat([prei[1] for prei in intv_info.pre_ac_intvs]...)
pre_b = vcat([prei[2] for prei in intv_info.pre_ac_intvs]...)
qcsector_info = QcSectorInfo(qxdim=qxdim, tband=1, pre_a=pre_a, pre_b=pre_b, base_a=0.0, base_b=1.0)

# qcinfos = [qcbounded_info]
# qcinfos = [qcsector_info]
qcinfos = [qcbounded_info, qcsector_info]


deepsdp_opts = DeepSdpOptions(verbose=true)


normal = [1.0; 1.0]
reach_soln = solveHplaneReach(nnet, input, qcinfos, deepsdp_opts, normal, verbose=true)
@printf("%s\n", reach_soln)

safety_soln = solveSafetyL2Gain(nnet, input, qcinfos, deepsdp_opts, 1000.0, verbose=true)
@printf("%s\n", safety_soln)

