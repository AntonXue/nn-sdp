using PyCall

# Load directly from an NNet file
function loadFromNnet(nnet_file::String, activ::Activ = ReluActiv())
  nnet = NNet(nnet_file)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  ffnet = FeedFwdNet(activ=activ, xdims=nnet.layerSizes, Ms=Ms)
  return ffnet
end

# Convert an ONNX to an NNet file
function onnx2nnet(onnx_file::String, nnet_file::String)
  # Load at path/to/EXTS_DIR/NNet
  pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)
  nnet = pyimport("NNet")
  use_gz = split(onnx_file, ".")[end] == "gz"
  if use_gz
      onnx_file = onnx_file[1:end-3]
  end
  nnet.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

# Load an ONNX file by first converting it to a NNet file
function loadFromOnnx(onnx_file::String, activ::Activ = ReluActiv())
  nnet_file = tempname()
  onnx2nnet(onnx_file, nnet_file)
  return loadFromNnet(nnet_file, activ)
end

# Check the extension and load accordingly
function loadFromFile(file, activ::Activ = ReluActiv())
  ext = split(file, ".")[end]
  if ext == "nnet"
    return loadFromNnet(file)
  elseif ext == "onnx"
    return loadFromOnnx(file)
  else
    error("Unrecognized file: $(file)")
  end
end

