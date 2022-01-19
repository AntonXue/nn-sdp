#
start_time = time()

#
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

println("Done importing stuff: " * string(round(time() - start_time, digits=2)))

# Seed is fixed, but all rand calls should also happen in the same expected sequence
Random.seed!(1234)

# Argument parsing
argparse_settings = ArgParseSettings()
@add_arg_table argparse_settings begin
      "--benchdir"
      help = "The directory of the benchmark"
      arg_type = String
end

args = parse_args(ARGS, argparse_settings)

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
  println("processing: " * nnet_filepath)
  nnet = NNetParser.NNet(nnet_filepath)
  ffnet = Utils.NNet2FeedForwardNetwork(nnet)

  image_filepath = joinpath(image_dir, nnet_filename * ".png")
  x1min = ones(nnet.inputSize) .- 1e-2
  x1max = ones(nnet.inputSize) .+ 1e-2
  runAndPlotRandomTrajectories(10000, ffnet, x1min, x1max, imgfile=image_filepath)
  println("\tdone: " * string(round(time() - loop_start_time, digits=2)))
end

println("end time: " * string(round(time() - start_time, digits=2)))


