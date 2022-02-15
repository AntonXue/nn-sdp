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

qxdim = sum(nnet.xdims[2:end-1])
acmin = vcat([xi[1] for xi in intv_info.x_intvs[2:end-1]]...)
acmax = vcat([xi[2] for xi in intv_info.x_intvs[2:end-1]]...)
qcbounded_info = QcBoundedInfo(qxdim=qxdim, acmin=acmin, acmax=acmax)
qcinfos = [qcbounded_info]

deepsdp_opts = DeepSdpOptions(verbose=true)

soln = solveSafetyL2Gain(nnet, input, qcinfos, deepsdp_opts, 1000.0, verbose=true)

