using LinearAlgebra
using ArgParse
using Printf
using Dates
using PyCall

using Plots
using Random
Random.seed!(1234)


include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "reach")
RAND_DIR = joinpath(@__DIR__, "..", "bench", "rand")
ind2reach(width,depth) = joinpath(RAND_DIR, "reach-I2-O2-W$(width)-D$(depth).nnet")

REACH_MOSEK_OPTS = 
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "MSK_IPAR_INTPNT_SCALING" => 1,
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-8,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

DOPTS = DeepSdpOptions(use_dual=true, mosek_opts=REACH_MOSEK_OPTS, verbose=true)
C2OPTS = ChordalSdpOptions(mosek_opts=REACH_MOSEK_OPTS, verbose=true, decomp_mode=DoubleDecomp())

function solveAndPlotEllipses(network_file::String, x1min::Vector, x1max::Vector, βs::VecInt, opts;
                              legend_title="",
                              saveto = joinpath(DUMP_DIR, "reach-" * basename(network_file) * ".png"))
  @assert length(βs) >= 1
  ffnet = loadFromFile(network_file)
  xfs = Utils.sampleTrajs(ffnet, x1min, x1max)
  xfs2D = [xf[1:2] for xf in xfs]

  trips = [findEllipsoid(ffnet, x1min, x1max, β, opts) for β in βs]
  ellipses = [(P, yc) for (P, yc, _) in trips]
  ellipses2D = [(P[1:2,1:2], yc[1:2]) for (P, yc) in trips]
  plt = plot()
  plt = Utils.plotBoundingEllipses!(plt, xfs2D, ellipses2D)

  β_strs = ["β = $(β)" for β in βs]
  labels = ["sampled"; β_strs]
  for (i, lbl) in enumerate(labels)
    plt[1][i][:label] = lbl
  end

  # plt = plot!(plt, legendtitle=legend_title)
  plt = plot!(plt, xtickfontsize=12, ytickfontsize=12, legendfontsize=12)
  savefig(plt, saveto)
  println("saved to: $(saveto)")
  return plt, trips
end

x1min, x1max = ones(2) .- 5e-1, ones(2) .+ 5e-1

βs = [0, 2, 4, 6, 8, 10]

# trips = solveAndPlotEllipses(args["nnet"], x1min, x1max, βs, DOPTS)

plt1, trips1 = solveAndPlotEllipses(ind2reach(10,10), x1min, x1max, βs, C2OPTS, legend_title="W10-D10")
plt2, trips2 = solveAndPlotEllipses(ind2reach(20,10), x1min, x1max, βs, C2OPTS, legend_title="W20-D10")

#=
solveAndPlotEllipses(ind2acas(1,2), x1min, x1max, βs, DOPTS)
solveAndPlotEllipses(ind2acas(1,3), x1min, x1max, βs, DOPTS)
solveAndPlotEllipses(ind2acas(1,4), x1min, x1max, βs, DOPTS)
=#



