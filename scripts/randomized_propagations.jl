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
rx_intvs, _, rslope_intvs = randomizedPropagation(input.x1min, input.x1max, ffnet, 100000)

# Set up the different options
split_opts1 = SplitSdpOptions(β=1, x_intvs=x_intvs, slope_intvs=slope_intvs, verbose=false)
rsplit_opts1 = SplitSdpOptions(β=1, x_intvs=rx_intvs, slope_intvs=rslope_intvs, verbose=false)

# The solves
split_poly1 = solveReachPolytope(ffnet, input, split_opts1, 6, verbose=true)
rsplit_poly1 = solveReachPolytope(ffnet, input, rsplit_opts1, 6, verbose=true)

# Do some random trajectories
xfs = randomTrajectories(10000, ffnet, input.x1min, input.x1max)

labeled_polys =
  [
    ("worst-case", split_poly1),
    ("randomized", rsplit_poly1),
  ]

# Save stuff
saveto_filepath = joinpath(args["odir"], "randprop-" * basename(nnet_filepath) * ".png")
plt = plotBoundingPolytopes(xfs, labeled_polys, saveto=saveto_filepath)


