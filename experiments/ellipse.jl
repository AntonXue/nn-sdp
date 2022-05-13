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
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)


function solveAndPlotEllipses(network_file::String, x1min::Vector, x1max::Vector, βs::VecInt, saveto=joinpath(DUMP_DIR, "reach-" * basename(network_file) * ".png"))
  @assert length(βs) >= 1
  ffnet = loadFromFile(network_file)
  xfs = Utils.sampleTrajs(ffnet, x1min, x1max)

  copts = ChordalSdpOptions(mosek_opts=ELLIPSE_MOSEK_OPTS, verbose=true, decomp_mode=TwoStage())
  trips = [findEllipsoid(ffnet, x1min, x1max, copts, β) for β in βs]
  ellipses = [(P, yc) for (P, yc, _) in trips]
  plt = plot()
  plt = Utils.plotBoundingEllipses!(plt, xfs, ellipses)
  savefig(plt, saveto)
  println("saved to: $(saveto)")
  return trips
end


x1min, x1max = ones(2) .- 1e-2, ones(2) .+ 1e-2

βs = [0, 2, 4, 6, 8, 10]

trips = solveAndPlotEllipses(args["nnet"], x1min, x1max, βs)

