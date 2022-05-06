start_time = time()

using LinearAlgebra
using ArgParse
using Printf
using Dates
using PyCall

using Plots

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

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
@printf("load done: %.3f\n", time() - start_time)

mosek_opts = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

ffnet = loadFromNnet(args["nnet"], ReluActiv())
# ffnet, αs = loadFromFileReluScaled(args["nnet"])
chordalsdp_opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=TwoStage())
deepsdp_opts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)

x1min, x1max = ones(2) .- 5e-1, ones(2) .+ 5e-1

xfs = Utils.sampleTrajs(ffnet, x1min, x1max)

P0, y0, soln0 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 0)
P2, _, soln2 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 2)
P4, _, soln4 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 4)
P6, _, soln6 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 6)
P8, _, soln8 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 8)

ellipses = [(P0, y0),
            (P2, y0),
            (P4, y0),
            (P6, y0),
            (P8, y0)]

plt = plot()
plt = Utils.plotBoundingEllipses!(plt, xfs, ellipses)

savefig(plt, "/home/antonxue/dump/foo.png")

#=
circle2 = findCircle(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes_nosec, solns_nosec = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes0, solns0 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes2, solns2 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes4, solns4 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 4)

labeled_polys =
[ ("nosec", hplanes_nosec),
  ("β=0", hplanes0),
  ("β=2", hplanes2),
  ("β=4", hplanes4),
]

xintvs_sliced = Intervals.intervalsAutoLirpaSliced(x1min, x1max, ffnet).x_intvs
xintvs_oneshot = Intervals.intervalsAutoLirpaOneShot(x1min, x1max, ffnet).x_intvs

sliced_lbs = [lb for (lb, _) in xintvs_sliced]
sliced_ubs = [ub for (_, ub) in xintvs_sliced]

oneshot_lbs = [lb for (lb, _) in xintvs_oneshot]
oneshot_ubs = [ub for (_, ub) in xintvs_oneshot]

xfs = randomTrajectories(10000, ffnet, x1min, x1max)
plt = plotBoundingPolys(xfs, labeled_polys)
=#


#=
points = [randn(2) for k in 1:1000]

polys = [([1;0.0], 1.0),
         ([0;1.0], 1.0),
         ([-1;0.0], 1.0),
         ([0;-1.0], 1.0)]

Ps = [Matrix(Symmetric(randn(2,2)) + 3 * I) for k in 1:3]

ellipses = [(P, [0.0; 0.0]) for P in Ps]

plt = plot()
=#


