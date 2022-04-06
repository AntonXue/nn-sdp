start_time = time()

using Dates
using ArgParse

include("../src/NnSdp.jl"); using .NnSdp

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      arg_type = String
      help = "The NNet file to load"
    "--onnx"
      arg_type = String
      help = "The ONNX file to load"
    "--vnnlib"
      arg_type = String
      help = "The vnnlib specifications"
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

# Load the network
ffnet = nothing
if !(args["onnx"] isa Nothing); ffnet = loadFromOnnx(args["onnx"]); end
if !(args["nnet"] isa Nothing); ffnet = loadFromNnet(args["nnet"]); end
@assert !(ffnet isa Nothing)

if haskey(args, "vnnlib"); spec = loadVnnlib(args["vnnlib"], ffnet) end

qc_inputs = [c[1] for c in spec]
qc_safeties = [c[2] for c in spec]

println("elapsed: $(time() - start_time)")
println("now: $(now())")

for (qc_input, qc_safety) in spec
  
end


