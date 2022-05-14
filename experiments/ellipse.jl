using LinearAlgebra
using ArgParse
using Printf
using Dates
using PyCall

using Plots


include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

DUMP_DIR = joinpath(@__DIR__, "..", "dump")
RAND_DIR = joinpath(@__DIR__, "..", "bench", "rand")


function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "the NNet file location"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

ELLIPSE_MOSEK_OPTS = 
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
        "MSK_IPAR_INTPNT_SCALING" => 2,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

DOPTS = DeepSdpOptions(use_dual=true, mosek_opts=ELLIPSE_MOSEK_OPTS, verbose=true)
COPTS = ChordalSdpOptions(mosek_opts=ELLIPSE_MOSEK_OPTS, verbose=true, decomp_mode=TwoStage())

function solveAndPlotEllipses(network_file::String, x1min::Vector, x1max::Vector, βs::VecInt, opts, saveto=joinpath(DUMP_DIR, "reach-" * basename(network_file) * ".png"))
  @assert length(βs) >= 1
  ffnet = loadFromFile(network_file)
  xfs = Utils.sampleTrajs(ffnet, x1min, x1max)
  xfs2D = [xf[1:2] for xf in xfs]

  trips = [findEllipsoid(ffnet, x1min, x1max, opts, β) for β in βs]
  ellipses = [(P, yc) for (P, yc, _) in trips]
  ellipses2D = [(P[1:2,1:2], yc[1:2]) for (P, yc) in trips]
  plt = plot()
  plt = Utils.plotBoundingEllipses!(plt, xfs2D, ellipses2D)
  savefig(plt, saveto)
  println("saved to: $(saveto)")
  return trips
end

# x1min, x1max = ones(2) .- 1e-2, ones(2) .+ 1e-2
x1min, x1max = ones(5) .- 1e-1, ones(5) .+ 1e-1

βs = [0, 2, 4, 6, 8]

trips = solveAndPlotEllipses(args["nnet"], x1min, x1max, βs, DOPTS)

