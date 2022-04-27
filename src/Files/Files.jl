# Interaction for methods of storing FeedFwdNet
module Files

using ..MyLinearAlgebra
using ..MyNeuralNetwork

EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")

# Load stuff
include(joinpath(EXTS_DIR, "nnet_parser.jl"))
include("network_files.jl");

export loadFromNnet, loadFromOnnx, loadFromFile
export onnx2nnet, nnet2onnx
export writeNnet, writeOnnx

end
