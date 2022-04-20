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
  if !(EXTS_DIR in PyVector(pyimport("sys")."path"))
    pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)
  end
  nnet = pyimport("NNet")
  use_gz = split(onnx_file, ".")[end] == "gz"
  if use_gz
    onnx_file = onnx_file[1:end-3]
  end
  nnet.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

# Convert from NNet to ONNX file
function nnet2onnx(nnet_file::String, onnx_file::String)
  if !(EXTS_DIR in PyVector(pyimport("sys")."path"))
    pushfirst!(PyVector(pyimport("sys")."path"), EXTS_DIR)
  end
  nnet = pyimport("NNet")
  nnet.nnet2onnx(nnet_file, onnxFile=onnx_file)
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

# Write FeedFwdNet to a NNet file
function writeNnet(ffnet::FeedFwdNet, saveto="$(homedir())/dump/hello.nnet")
  xdims = ffnet.xdims
  nnet_file = saveto
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

