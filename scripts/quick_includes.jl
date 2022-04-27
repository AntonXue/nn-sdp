start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using Dates
using PyCall

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

mosek_opts = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true)
deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)

x1min, x1max = ones(2) .- 5e-1, ones(2) .+ 5e-1
hplanes_nosec, solns_nosec = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0, use_qc_sector=false)
hplanes0, solns0 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes2, solns2 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes4, solns4 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 4)
hplanes6, solns6 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 6)
hplanes8, solns8 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 8)
hplanes10, solns10 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 10)
hplanes12, solns12 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 12)

labeled_polys =
[ ("nosec", hplanes_nosec),
  ("β=0", hplanes0),
  ("β=2", hplanes2),
  ("β=4", hplanes4),
  ("β=6", hplanes6),
  ("β=8", hplanes8),
  ("β=10", hplanes10),
  ("β=12", hplanes12),
]

xfs = randomTrajectories(10000, ffnet, x1min, x1max)
plt = plotBoundingPolys(xfs, labeled_polys)

rand_nnet = "/home/antonxue/dump/rand.nnet"
rand_onnx = "/home/antonxue/dump/rand.onnx"
rand_ffnet = Utils.randomNetwork([2;3;4;5;4;3;2])
writeNnet(rand_ffnet, rand_nnet)
nnet2onnx(rand_nnet, rand_onnx)

EXTS_DIR = "/home/antonxue/nn-sdp/exts"
pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)

bridge = pyimport("auto_lirpa_bridge")

