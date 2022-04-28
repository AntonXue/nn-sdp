# Interaction for methods of storing FeedFwdNet
module Files

using ..MyLinearAlgebra
using ..MyNeuralNetwork

# Load stuff
include("network_files.jl");

export loadFromNnet, loadFromOnnx, loadFromFile
export onnx2nnet, nnet2onnx
export writeNnet, writeOnnx

end

