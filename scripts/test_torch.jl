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
SHAORU1_PATH = joinpath(@__DIR__, "..", "models", "shaoru1.pth")
SHAORU2_PATH = joinpath(@__DIR__, "..", "models", "shaoru2.pth")
SHAORU3_PATH = joinpath(@__DIR__, "..", "models", "shaoru3.pth")
SHAORU4_PATH = joinpath(@__DIR__, "..", "models", "shaoru4.pth")

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
shaoru1_ffnet = load(SHAORU1_PATH)
shaoru2_ffnet = load(SHAORU2_PATH)
shaoru3_ffnet = load(SHAORU3_PATH)
shaoru4_ffnet = load(SHAORU4_PATH)

x1min, x1max = -1e-2 * ones(4), 1e-2 * ones(4)
boxes1_beta0 = nn.findReachBox(shaoru1_ffnet, x1min, x1max, 0, DOPTS)
printstyled("BOXES 1 BETA 0:\n", color=:green)
for b in boxes1_beta0; println(b) end

# boxes1_beta1 = nn.findReachBox(shaoru1_ffnet, x1min, x1max, 1, C2OPTS)
# printstyled("BOXES 1 BETA 1:\n", color=:green)
# for b in boxes1_beta1; println(b) end

boxes2_beta0 = nn.findReachBox(shaoru2_ffnet, x1min, x1max, 0, DOPTS)
printstyled("BOXES 2 BETA 0:\n", color=:green)
for b in boxes2_beta0; println(b) end

# boxes2_beta1 = nn.findReachBox(shaoru2_ffnet, x1min, x1max, 1, C2OPTS)
# printstyled("BOXES 2 BETA 1:\n", color=:green)
# for b in boxes2_beta1; println(b) end

boxes3_beta0 = nn.findReachBox(shaoru3_ffnet, x1min, x1max, 0, DOPTS)
printstyled("BOXES 2 BETA 0:\n", color=:green)
for b in boxes3_beta0; println(b) end

# boxes3_beta1 = nn.findReachBox(shaoru3_ffnet, x1min, x1max, 1, C2OPTS)
# printstyled("BOXES 2 BETA 1:\n", color=:green)
# for b in boxes3_beta1; println(b) end

boxes4_beta0 = nn.findReachBox(shaoru4_ffnet, x1min, x1max, 0, DOPTS)
printstyled("BOXES 2 BETA 0:\n", color=:green)
for b in boxes4_beta0; println(b) end

# boxes4_beta1 = nn.findReachBox(shaoru4_ffnet, x1min, x1max, 1, C2OPTS)
# printstyled("BOXES 2 BETA 1:\n", color=:green)
# for b in boxes4_beta1; println(b) end

boxes5_beta0 = nn.findReachBox(shaoru5_ffnet, x1min, x1max, 0, DOPTS)
printstyled("BOXES 2 BETA 0:\n", color=:green)
for b in boxes5_beta0; println(b) end

# boxes5_beta1 = nn.findReachBox(shaoru5_ffnet, x1min, x1max, 1, C2OPTS)
# printstyled("BOXES 2 BETA 1:\n", color=:green)
# for b in boxes5_beta1; println(b) end



