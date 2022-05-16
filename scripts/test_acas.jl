start_time = time()

using LinearAlgebra
using Dates
using ArgParse

include("../src/NnSdp.jl"); using .NnSdp
include("../experiments/vnnlib_utils.jl")

# The place where things are
DUMP_DIR = joinpath(@__DIR__, "..", "dump", "acas")
ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")

# The ACAS files
ind2acas(i,j) = joinpath(ACAS_DIR, "ACASXU_run2a_$(i)_$(j)_batch_2000.onnx")
ACAS_FILES = [ind2acas(i,j) for i in 1:5 for j in 1:9]
@assert length(ACAS_FILES) == 45

# The spec files
ind2spec(i) = joinpath(ACAS_DIR, "prop_$(i).vnnlib")
SPEC_FILES = [ind2spec(i) for i in 1:10]
@assert length(SPEC_FILES) == 10

MOSEK_OPTS =
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 30, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)


DOPTS = DeepSdpOptions(use_dual=true, verbose=true, mosek_opts=MOSEK_OPTS)
COPTS = ChordalSdpOptions(verbose=true, mosek_opts=MOSEK_OPTS, decomp_mode=SingleDecomp())
C2OPTS = ChordalSdpOptions(verbose=true, mosek_opts=MOSEK_OPTS, decomp_mode=DoubleDecomp())


cnf_qs = loadReluQueriesCnf(ind2acas(1,9), ind2spec(7), 3)
q11 = cnf_qs[1][1]

