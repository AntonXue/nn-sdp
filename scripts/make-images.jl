start_time = time()
include("../src/core/header.jl"); using .Header
include("../src/core/common.jl"); using .Common
include("../src/parsers/nnet-parser.jl"); using .NNetParser
include("../src/parsers/vnnlib-parser.jl"); using .VnnlibParser
include("../src/utils.jl"); using .Utils
include("../src/tests.jl"); using .Tests

using LinearAlgebra
using JuMP
using Random
using ArgParse
using Printf

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--benchdir"
      help = "The directory of the benchmark"
      arg_type = String
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

# Set up the benchdir and image dir
@assert isdir(args["benchdir"])
nnet_dir = joinpath(args["benchdir"], "nnet")
@assert isdir(nnet_dir)

image_dir = joinpath(args["benchdir"], "image")
if !isdir(image_dir)
  mkdir(image_dir)
end

# Go through each nnet and plot their image
for nnet_filename in readdir(nnet_dir, join=false)
  loop_start_time = time()
  nnet_filepath = joinpath(nnet_dir, nnet_filename)
  @printf("processing: %s\n", nnet_filepath)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)

  image_filepath = joinpath(image_dir, nnet_filename * ".png")
  x1min = ones(nnet.inputSize) .- 1e-2
  x1max = ones(nnet.inputSize) .+ 1e-2
  plotRandomTrajectories(10000, ffnet, x1min, x1max, saveto=image_filepath)
  @printf("\tdone: %.3f\n", time() - loop_start_time)
end

@printf("end time: %.3f\n", time() - start_time)


