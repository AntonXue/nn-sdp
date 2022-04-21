module Files

using ..MyLinearAlgebra
using ..MyNeuralNetwork
# using ..Qc

EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")

# Load stuff
include(joinpath(EXTS_DIR, "nnet_parser.jl"))
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))
include("network_files.jl");
# include("query_files.jl");

export loadFromNnet, loadFromOnnx, loadFromFile, loadVnnlib
export onnx2nnet, nnet2onnx
export writeNnet, writeOnnx
# export loadQueries

end
