start_time = time()

include("../src/core/header.jl"); using .Header
include("../src/core/common.jl"); using .Common
include("../src/core/intervals.jl"); using .Intervals
include("../src/core/deep-sdp.jl"); using .DeepSdp
include("../src/core/split-sdp.jl"); using .SplitSdp
include("../src/parsers/nnet-parser.jl"); using .NNetParser
include("../src/utils.jl"); using .Utils
include("../src/methods.jl"); using .Methods

using LinearAlgebra
using JuMP
using Random
using ArgParse
using Printf

# Fix
Random.seed!(1234)

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "The NNet file we are making bounding boxes of"
      arg_type = String
      required = true

    "--odir"
      help = "The output directory"
      arg_type = String
      default = "~/Desktop"
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()
nnet_filepath = args["nnet"]
nnet = NNetParser.NNet(nnet_filepath)
ffnet = Utils.NNet2FeedForwardNetwork(nnet)

x1min = ones(nnet.inputSize) .- 1e-2
x1max = ones(nnet.inputSize) .+ 1e-2
input = BoxInput(x1min=x1min, x1max=x1max)

# Do some interval propagation
x_intvs, _, slope_intvs = worstCasePropagation(input.x1min, input.x1max, ffnet)

# Set up the different options
deep_opts = DeepSdpOptions(x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
split_opts1 = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
split_opts2 = SplitSdpOptions(β=2, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
split_opts3 = SplitSdpOptions(β=3, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
split_opts3 = SplitSdpOptions(β=4, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
split_opts3 = SplitSdpOptions(β=5, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)

# The solves
deep_poly = solveReachPolytope(ffnet, input, deep_opts, 6, verbose=true)
split_poly1 = solveReachPolytope(ffnet, input, split_opts1, 6, verbose=true)
split_poly2 = solveReachPolytope(ffnet, input, split_opts2, 6, verbose=true)
# split_poly3 = solveReachPolytope(ffnet, input, split_opts3, 6, verbose=true)
# split_poly4 = solveReachPolytope(ffnet, input, split_opts4, 6, verbose=true)
# split_poly5 = solveReachPolytope(ffnet, input, split_opts5, 6, verbose=true)

# Do some random trajectories
xfs = randomTrajectories(10000, ffnet, input.x1min, input.x1max)

labeled_polys =
  [
    ("deepsdp", deep_poly),
    ("β=1", split_poly1),
    ("β=2", split_poly2),
    # ("β=3", split_poly3),
    # ("β=4", split_poly4),
    # ("β=5", split_poly5),
  ]

# Save stuff
saveto_filepath = joinpath(args["odir"], "bound-" * basename(nnet_filepath) * ".png")
@printf("saving to: %s\n", saveto_filepath)
plt = plotBoundingPolytopes(xfs, labeled_polys, saveto=saveto_filepath)


