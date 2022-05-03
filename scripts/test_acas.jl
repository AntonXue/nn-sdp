start_time = time()

using LinearAlgebra
using Dates
using ArgParse

include("../src/NnSdp.jl"); using .NnSdp

ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")
ALL_FILES = readdir(ACAS_DIR, join=true)
ACAS_FILES = filter(f -> match(r".*.onnx", f) isa RegexMatch, ALL_FILES)
SPEC_FILES = filter(f -> match(r".*.vnnlib", f) isa RegexMatch, ALL_FILES)

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


mosek_opts = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)


deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts)
chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, two_stage=true)

# loadQueries(ACAS_FILES[1], SPEC_FILES[1], β)

β = 2

dnf = loadReluQueries(args["onnx"], args["vnnlib"], β)

qs = vcat(dnf...)






