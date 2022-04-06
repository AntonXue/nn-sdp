using ArgParse
using PyCall
using Random
Random.seed!(1234)

# Set up the path
NNET_PATH = joinpath(@__DIR__, "..", "src", "Utils", "stolen_code")
pushfirst!(PyVector(pyimport("sys")."path"), NNET_PATH)
NNet = pyimport("NNet")

INPUT_DIM = 2
OUTPUT_DIM = 2
LAYER_DIMS = [5; 10; 15; 20]
NUM_LAYERS = 5:5:50

# Generate some randomized parameters
function randomParams(input_dim, output_dim, layer_dim, num_layers, σ)
  xdims = Int.([input_dim; ones(num_layers) * layer_dim; output_dim])
  Ws = [σ * randn(xdims[k+1], xdims[k]) for k in 1:length(xdims)-1]
  bs = [σ * randn(xdims[k+1]) for k in 1:length(xdims)-1]
  return xdims, Ws, bs
end

# Give all the random parameters in a list
function makeRandomParams()
  params = Vector{Any}()
  for layer_dim in LAYER_DIMS
    for num_layers in NUM_LAYERS
      σ = 2 / sqrt(layer_dim + num_layers)
      input_dim = INPUT_DIM
      output_dim = OUTPUT_DIM
      xdims, Ws, bs = randomParams(input_dim, output_dim, layer_dim, num_layers, σ)
      push!(params, (input_dim, output_dim, layer_dim, num_layers, xdims, Ws, bs))
    end
  end
  return params
end

# Do some writes
function writeNNet(input_dim, Ws, bs, file)
  input_mins = -10000 * ones(input_dim)
  input_maxes = 10000 * ones(input_dim)
  means = zeros(input_dim + 1)
  ranges = ones(input_dim + 1)
  NNet.utils.writeNNet.writeNNet(Ws, bs, input_mins, input_maxes, means, ranges, file)
end

# Command line arguments
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnetdir"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@assert isdir(args["nnetdir"])

params = makeRandomParams()
num_params = length(params)

for (i, (input_dim, output_dim, layer_dim, num_layers, _, Ws, bs)) in enumerate(params)
  idim, odim, ldim, numl = input_dim, output_dim, layer_dim, num_layers
  file = "rand-I$(idim)-O$(odim)-W$(ldim)-D$(numl).nnet"
  file = joinpath(args["nnetdir"], file)
  println("writing [$(i)/$(num_params)]: $(file)")
  writeNNet(idim, Ws, bs, file)
end

