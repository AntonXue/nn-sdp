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
       "INTPNT_CO_TOL_REL_GAP" => 1e-8,
       "INTPNT_CO_TOL_PFEAS" => 1e-8,
       "INTPNT_CO_TOL_DFEAS" => 1e-8)


deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts)
chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, two_stage_cliques=true)

# loadQueries(ACAS_FILES[1], SPEC_FILES[1], β)

β = 2

queries = loadQueries(args["onnx"], args["vnnlib"], β)

#=
q1s = loadQueries(ACAS_FILES[1], SPEC_FILES[1], β)
q2s = loadQueries(ACAS_FILES[1], SPEC_FILES[3], β)
q3s = loadQueries(ACAS_FILES[1], SPEC_FILES[4], β)
q4s = loadQueries(ACAS_FILES[1], SPEC_FILES[5], β)
q5s = loadQueries(ACAS_FILES[1], SPEC_FILES[6], β)
q6s = loadQueries(ACAS_FILES[1], SPEC_FILES[7], β)
q7s = loadQueries(ACAS_FILES[1], SPEC_FILES[8], β)
q8s = loadQueries(ACAS_FILES[1], SPEC_FILES[9], β)
q9s = loadQueries(ACAS_FILES[1], SPEC_FILES[10], β)
=#


