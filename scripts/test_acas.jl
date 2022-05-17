start_time = time()

using LinearAlgebra
using Dates
using ArgParse
using MosekTools


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
       "MSK_IPAR_INTPNT_SCALING" => 2,
       "INTPNT_CO_TOL_REL_GAP" => 1e-9,
       "INTPNT_CO_TOL_PFEAS" => 1e-9,
       "INTPNT_CO_TOL_DFEAS" => 1e-9)


DOPTS = DeepSdpOptions(use_dual=true, verbose=true, mosek_opts=MOSEK_OPTS)
COPTS = ChordalSdpOptions(verbose=true, mosek_opts=MOSEK_OPTS, decomp_mode=SingleDecomp())
C2OPTS = ChordalSdpOptions(use_dual=true, verbose=true, mosek_opts=MOSEK_OPTS, decomp_mode=DoubleDecomp())


cnf_qs = loadReluQueriesCnf(ind2acas(1,9), ind2spec(7), 3)
q11 = cnf_qs[1][1]


dsoln = NnSdp.solveQuery(q11, DOPTS)
dγin = dsoln.values[:γin]
dγac1 = dsoln.values[:γac1]
dγac2 = dsoln.values[:γac2]

dZin = makeZin(dγin, q11.qc_input, q11.ffnet)
dZac1 = makeZac(dγac1, q11.qc_activs[1], q11.ffnet)
dZac2 = makeZac(dγac2, q11.qc_activs[2], q11.ffnet)
dZout = makeZout(q11.qc_safety, q11.ffnet)
dZ = dZin + dZac1 + dZac2 + dZout

#################

c2soln = NnSdp.solveQuery(q11, C2OPTS)
c2γin = c2soln.values[:γin]
c2γac1 = c2soln.values[:γac1]
c2γac2 = c2soln.values[:γac2]

c2Zin = makeZin(c2γin, q11.qc_input, q11.ffnet)
c2Zac1 = makeZac(c2γac1, q11.qc_activs[1], q11.ffnet)
c2Zac2 = makeZac(c2γac2, q11.qc_activs[2], q11.ffnet)
c2Zout = makeZout(q11.qc_safety, q11.ffnet)

c2Z = c2Zin + c2Zac1 + c2Zac2 + c2Zout


