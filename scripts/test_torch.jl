start_time = time()

using LinearAlgebra
using Printf
using Dates
using Plots

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

@printf("load done: %.3f\n", time() - start_time)


CTRL_PATH = joinpath(@__DIR__, "..", "models", "ctrl.pth")
CART_PATH = joinpath(@__DIR__, "..", "models", "cart.pth")
SHAORU_PATH = joinpath(@__DIR__, "..", "models", "shaoru.pth")

mosek_opts = 
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

DOPTS = DeepSdpOptions(use_dual=true, mosek_opts=mosek_opts, verbose=true)
COPTS = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:single_decomp)
C2OPTS = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:double_decomp)

# cart_ffnet = load(CART_PATH)
# ctrl_ffnet = load(CTRL_PATH)
shaoru_ffnet = load(SHAORU_PATH)

