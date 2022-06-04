using ArgParse
using PyCall
using Random
Random.seed!(1234)

# Set up the path
NNET_PATH = joinpath(@__DIR__, "..", "exts") 
pushfirst!(PyVector(pyimport("sys")."path"), NNET_PATH)
NNet = pyimport("NNet")

SCALE_IN_DIM = 2
SCALE_OUT_DIM = 2
SCALE_ALL_LAYER_DIMS = 5:5:50
SCALE_ALL_NUM_LAYERS = 5:5:100

REACH_IN_DIM = 2
REACH_OUT_DIM = 2
REACH_ALL_LAYER_DIMS = [10, 20, 30]
REACH_ALL_NUM_LAYERS = [10, 20, 30]

# Generate some randomized parameters
function randomParams(in_dim, out_dim, layer_dim, num_layers, σ)
  xdims = Int.([in_dim; ones(num_layers) * layer_dim; out_dim])
  Ws = [σ * randn(xdims[k+1], xdims[k]) for k in 1:length(xdims)-1]
  bs = [σ * randn(xdims[k+1]) for k in 1:length(xdims)-1]
  return xdims, Ws, bs
end

# Give all the parameters in a list
function makeParams(in_dim, out_dim, all_layer_dims, all_num_layers, σ::Function)
  params = Vector{Any}()
  for layer_dim in all_layer_dims
    for num_layers in all_num_layers
      thisσ = σ(layer_dim, num_layers)
      xdims, Ws, bs = randomParams(in_dim, out_dim, layer_dim, num_layers, thisσ)
      push!(params, (in_dim, out_dim, layer_dim, num_layers, xdims, Ws, bs))
    end
  end
  return params
end

# The parameters for scaling experiments
function makeScaleParams()
  σ(layer_dim, _) = 2 / sqrt(layer_dim * log(layer_dim))
  return makeParams(SCALE_IN_DIM, SCALE_OUT_DIM, SCALE_ALL_LAYER_DIMS, SCALE_ALL_NUM_LAYERS, σ)
end

# The parameters for reach experiments
function makeReachParams()
  σ(_, _) = 1 / sqrt(2)
  return makeParams(REACH_IN_DIM, REACH_OUT_DIM, REACH_ALL_LAYER_DIMS, REACH_ALL_NUM_LAYERS, σ)
end

# Do some writes
function writeNNet(in_dim, Ws, bs, file)
  input_mins = -10000 * ones(in_dim)
  input_maxes = 10000 * ones(in_dim)
  means = zeros(in_dim + 1)
  ranges = ones(in_dim + 1)
  NNet.utils.writeNNet.writeNNet(Ws, bs, input_mins, input_maxes, means, ranges, file)
end

# Command line arguments
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnetdir"
      arg_type = String
      default = joinpath(@__DIR__, "..", "bench", "rand")
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@assert isdir(args["nnetdir"])

# Generate the scalability networks
scale_params = makeScaleParams()
num_scale_params = length(scale_params)

for (i, (in_dim, out_dim, layer_dim, num_layers, _, Ws, bs)) in enumerate(scale_params)
  idim, odim, ldim, numl = in_dim, out_dim, layer_dim, num_layers
  file = "scale-I$(idim)-O$(odim)-W$(ldim)-D$(numl).nnet"
  file = joinpath(args["nnetdir"], file)
  println("writing [$(i)/$(length(scale_params))]: $(file)")
  writeNNet(idim, Ws, bs, file)
end

# Generate the reachability networks

reach_params = makeReachParams()
num_reach_params = length(reach_params)

for (i, (in_dim, out_dim, layer_dim, num_layers, _, Ws, bs)) in enumerate(reach_params)
  idim, odim, ldim, numl = in_dim, out_dim, layer_dim, num_layers
  file = "reach-I$(idim)-O$(odim)-W$(ldim)-D$(numl).nnet"
  file = joinpath(args["nnetdir"], file)
  println("writing [$(i)/$(length(reach_params))]: $(file)")
  writeNNet(idim, Ws, bs, file)
end


