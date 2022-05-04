# File IO for neural network
using LinearAlgebra
using PyCall

# Load a dependency
NNET_PARENT_DIR = joinpath(@__DIR__, "..", "..", "exts")
include(joinpath(NNET_PARENT_DIR, "nnet_parser.jl"))

# Set p bridge to NNet
if !(NNET_PARENT_DIR in PyVector(pyimport("sys")."path"))
  pushfirst!(PyVector(pyimport("sys")."path"), NNET_PARENT_DIR)
end

nnet_bridge = pyimport("NNet")

# Load directly from an NNet file
function loadFromNnet(nnet_file::String, activ::Activ = ReluActiv())
  nnet = NNet(nnet_file)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  ffnet = FeedFwdNet(activ=activ, xdims=nnet.layerSizes, Ms=Ms)
  return ffnet
end

# Convert an ONNX to an NNet file
# A NNet file assumes everything is a relu
function onnx2nnet(onnx_file::String, nnet_file::String)
  use_gz = split(onnx_file, ".")[end] == "gz"
  if use_gz
    onnx_file = onnx_file[1:end-3]
  end
  nnet_bridge.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

# Convert from NNet to ONNX file
# A NNet file does not have activation info, so we explicitly supply it
function nnet2onnx(nnet_file::String, onnx_file::String, activ::Activ)
  # These need to match the ONNX operator naming conventions
  if activ isa ReluActiv
    activ_str = "Relu"
  elseif activ isa TanhActiv
    activ_str = "Tanh"
  else
    error("unsupported activation: $(ffnet.activ)")
  end
  nnet_bridge.nnet2onnx(nnet_file, onnxFile=onnx_file, activ=activ_str)
end

# Load an ONNX file by first converting it to a NNet file
function loadFromOnnx(onnx_file::String, activ::Activ = ReluActiv())
  nnet_file = tempname()
  onnx2nnet(onnx_file, nnet_file)
  return loadFromNnet(nnet_file, activ)
end

# Check the extension and load accordingly
function loadFromFile(file::String, activ::Activ = ReluActiv())
  ext = split(file, ".")[end]
  if ext == "nnet"
    return loadFromNnet(file, activ)
  elseif ext == "onnx"
    return loadFromOnnx(file)
  else
    error("unrecognized file: $(file)")
  end
end

#= Load a relu network while scaling weights
We scale such that each new Wk has opnorm
  ||Wk|| = sqrt(ck * log ck / K), where ck = xdims[k] + xdims[k+1]

Use a sequence of α[1], ..., α[K] such that
  Wk -> αk Wk,    bk -> prod(αs[1:k]) bk

Let x' be an input to the scaled network f'
Observe tha: f'(x) = prod(αs) f(x)
Thus, f'(x') = f(x) iff x = prod(αs) x'

Again, this only works for piecewise-linear activations like relu
=#
function loadFromFileReluScaled(file::String)
  ffnet = loadFromFile(file, ReluActiv())
  xdims, Ms, K = ffnet.xdims, ffnet.Ms, ffnet.K
  Ws, bs = [M[:,1:end-1] for M in Ms], [M[:,end] for M in Ms]
  tgt_func(ck) = sqrt(ck * log(ck) / K)
  tgt_opnorms = [tgt_func(xdims[k]+xdims[k+1]) for k in 1:K]
  αs = [tgt_opnorms[k] / opnorm(W) for (k, W) in enumerate(Ws)]
  scaled_Ws = [αs[k] * Ws[k] for k in 1:K]
  scaled_bs = [prod(αs[1:k]) * bs[k] for k in 1:K]
  scaled_Ms = [[scaled_Ws[k] scaled_bs[k]] for k in 1:K]
  scaled_ffnet = FeedFwdNet(activ=ReluActiv(), xdims=xdims, Ms=scaled_Ms)
  return scaled_ffnet, αs
end

# Write FeedFwdNet to a NNet file
function writeNnet(ffnet::FeedFwdNet, nnet_file="$(homedir())/dump/hello.nnet")
  xdims = ffnet.xdims
  open(nnet_file, "w") do f
    # Component 1
    write(f, "// Dummy header\n")

    # Component 2
    num_layers, input_size, output_size, max_layer_size = ffnet.K, xdims[1], xdims[end], maximum(xdims)
    write(f, "$(num_layers), $(input_size), $(output_size), $(max_layer_size)\n")

    # Component 3
    write(f, "$(join(xdims, ","))\n")

    # Component 4
    write(f, "0\n")

    # Component 5
    input_min = -100000
    min_str = join(input_min * ones(input_size), ",")
    write(f, "$(min_str)\n")

    # Component 6
    input_max = -1 * input_min
    max_str = join(input_max * ones(input_size), ",")
    write(f, "$(max_str)\n")

    # Component 7
    mean_str = join(zeros(num_layers+1), ",")
    write(f, "$(mean_str)\n")

    # Component 8
    range_str = join(ones(num_layers+1), ",")
    write(f, "$(range_str)\n")

    # Component 9
    for k in 1:ffnet.K
      Mk, bk = ffnet.Ms[k][:, 1:end-1], ffnet.Ms[k][:, end]
      for i in 1:size(Mk)[1]
        for j in 1:size(Mk)[2]
          write(f, "$(Mk[i,j]),")
        end
        write(f, "\n")
      end

      for i in 1:length(bk)
        write(f, "$(bk[i]),\n")
      end
    end

    # Done
  end
end

# Write a FeedFwdNet to an ONNX file
function writeOnnx(ffnet::FeedFwdNet, onnx_file="$(homedir())/dump/hello.onnx")
  nnet_file = tempname()
  writeNnet(ffnet, nnet_file)
  nnet2onnx(nnet_file, onnx_file, ffnet.activ)
end

