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

mosek_opts = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true)
deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)

x1min, x1max = ones(2) .- 1e-2, ones(2) .+ 1e-2
hplanes0, solns0 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes1, solns1 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 1)
hplanes2, solns2 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes3, solns3 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 3)
hplanes4, solns4 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 4)
hplanes5, solns5 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 5)

labeled_polys =
  [ ("β=0", hplanes0),
    ("β=1", hplanes1),
    ("β=2", hplanes2),
    ("β=3", hplanes3),
    ("β=4", hplanes4),
    ("β=5", hplanes5),
  ]

xfs = randomTrajectories(10000, ffnet, x1min, x1max)
plt = plotBoundingPolys(xfs, labeled_polys)


