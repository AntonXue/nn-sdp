# File IO for neural network
using LinearAlgebra
using PyCall
using NaturalSort

# Load a dependency
NNET_PARENT_DIR = joinpath(@__DIR__, "..", "..", "exts")
MODELS_DIR = joinpath(@__DIR__, "..", "..", "models")
include(joinpath(NNET_PARENT_DIR, "nnet_parser.jl"))

# Set p bridge to NNet
if !(NNET_PARENT_DIR in PyVector(pyimport("sys")."path"))
  pushfirst!(PyVector(pyimport("sys")."path"), NNET_PARENT_DIR)
end

nnet_bridge = pyimport("NNet")

abstract type FileFormat end
struct NNetFormat <: FileFormat end
struct OnnxFormat <: FileFormat end
struct TorchFormat <: FileFormat end

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
  activ_str = activString(activ)
  nnet_bridge.nnet2onnx(nnet_file, onnxFile=onnx_file, activ=activ_str)
end

# Load from NNet
function load(::NNetFormat, file::String; activ = ReluActiv())
  nnet = NNet(file)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  ffnet = FeedFwdNet(activ=activ, xdims=nnet.layerSizes, Ms=Ms)
  return ffnet
end

# Load an ONNX file by first converting it to a NNet file
function load(::OnnxFormat, file::String; activ = ReluActiv())
  nnet_file = tempname()
  onnx2nnet(file, nnet_file)
  return load(NNetFormat(), nnet_file, activ)
end

# Load stuff from torch
function load(::TorchFormat, file::String; activ = ReluActiv())
  torch = pyimport("torch")
  d = torch.load(file, torch.device("cpu"))
  params = sort(collect(d), lt=(a,b) -> natural(a[1], b[1]))
  bs = map(kv -> kv[2].numpy()*1.0, filter(kv -> occursin("bias", kv[1]), params))
  Ws = map(kv -> kv[2].numpy()*1.0, filter(kv -> occursin("weight", kv[1]), params))
  @assert length(bs) == length(Ws)
  Ms = [[W b] for (W, b) in zip(Ws, bs)]
  xdims = [size(W)[2] for W in Ws]
  xdims = [xdims; size(bs[end])[1]]
  ffnet = FeedFwdNet(activ=activ, xdims=xdims, Ms=Ms)
  return ffnet
end

# Guess the extension
function load(file::String; activ=ReluActiv())
  ext = split(file, ".")[end]
  format = if ext == "nnet"; NNetFormat()
           elseif ext == "onnx"; OnnxFormat()
           elseif ext == "pt" || ext == "pth"; TorchFormat()
           else error("unrecognized: $(file)")
           end
  return load(format, file, activ=activ)
end

# The scaling mode that happens
abstract type ScalingMethod end
struct NoScaling <: ScalingMethod end
struct SqrtLogScaling <: ScalingMethod end
@with_kw struct FixedNormScaling <: ScalingMethod; Wk_opnorm::Real; end
@with_kw struct FixedConstScaling <: ScalingMethod; α::Real; end

#= Load a relu network while scaling weights
Use a sequence of α[1], ..., α[K] such that
  Wk -> αk Wk,    bk -> prod(αs[1:k]) bk

Let x' be an input to the scaled network f'
Observe that: f'(x) = prod(αs) f(x)

Again, this only works for piecewise-linear activations like relu
=#
function loadScaled(file::String, scaling_func::Function)
  ffnet = load(file, activ=ReluActiv())
  xdims, Ms, K = ffnet.xdims, ffnet.Ms, ffnet.K
  Ws, bs = [M[:,1:end-1] for M in Ms], [M[:,end] for M in Ms]
  αs = scaling_func(ffnet)
  scaled_Ws = [αs[k] * Ws[k] for k in 1:K]
  scaled_bs = [prod(αs[1:k]) * bs[k] for k in 1:K]
  scaled_Ms = [[scaled_Ws[k] scaled_bs[k]] for k in 1:K]
  scaled_ffnet = FeedFwdNet(activ=ReluActiv(), xdims=xdims, Ms=scaled_Ms)
  return scaled_ffnet, αs
end

scalingFunc(::NoScaling) = ffnet -> ones(ffnet.K)
scalingFunc(sc::FixedConstScaling) = ffnet -> sc.α * ones(ffnet.K)

function scalingFunc(::SqrtLogScaling)
  return ffnet ->
    let
      Ws, xdims, K = [M[:,1:end-1] for M in ffnet.Ms], ffnet.xdims, ffnet.K
      tgt_func(ck) = sqrt(ck * log(ck) / K)
      # tgt_func(ck) = 0.5 * sqrt(ck * log(ck) / K)
      # tgt_func(ck) = 0.3 * sqrt(ck * log(ck) / K)
      tgt_opnorms = [tgt_func(xdims[k]+xdims[k+1]) for k in 1:K]
      return [tgt_opnorms[k] / opnorm(W) for (k, W) in enumerate(Ws)]
    end
end

function scalingFunc(sc::FixedNormScaling)
  return ffnet ->
    let
      Ws = [M[:,1:end-1] for M in ffnet.Ms]
      return [sc.Wk_opnorm / opnorm(W) for W in Ws]
    end
end

# Call this
loadScaled(file, sc::ScalingMethod) = loadScaled(file, scalingFunc(sc))

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

