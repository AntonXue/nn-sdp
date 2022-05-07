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
x1min, x1max = ones(2) .- 5e-1, ones(2) .+ 5e-1


dopts = DeepSdpOptions(mosek_opts=mosek_opts, verbose=true)
copts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=OneStage())
c2opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=TwoStage())



# Hyperplane reach
qc_input = QcInputBox(x1min=x1min, x1max=x1max)
qc_activs = Qc.makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=2)
qc_reach = QcReachHplane(normal=[1; 1])
obj_func = x -> x[1]
reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_activs=qc_activs, qc_reach=qc_reach, obj_func=obj_func)

dsolnh = Methods.runQuery(reach_query, dopts)
csolnh = Methods.runQuery(reach_query, copts)
c2solnh = Methods.runQuery(reach_query, c2opts)


println("\n")

dsoln = findCircle(ffnet, x1min, x1max, dopts, 1)
csoln = findCircle(ffnet, x1min, x1max, copts, 1)
c2soln = findCircle(ffnet, x1min, x1max, c2opts, 1)

println("\n")

_, _, dsolne = findEllipsoid(ffnet, x1min, x1max, dopts, 1)
_, _, csolne = findEllipsoid(ffnet, x1min, x1max, copts, 1)
_, _, c2solne = findEllipsoid(ffnet, x1min, x1max, c2opts, 1)


#=
x1min, x1max = ones(2) .- 5e-1, ones(2) .+ 5e-1

xfs = Utils.sampleTrajs(ffnet, x1min, x1max)

P0, yc, soln0 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 0)
P2, _, soln2 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 2)
P4, _, soln4 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 4)
P6, _, soln6 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 6)
P8, _, soln8 = findEllipsoid(ffnet, x1min, x1max, chordalsdp_opts, 8)

ellipses = [(P0, yc),
            (P2, yc),
            (P4, yc),
            (P6, yc),
            (P8, yc)]

plt = plot()
plt = Utils.plotBoundingEllipses!(plt, xfs, ellipses)

saveto = joinpath(homedir(), "dump", "reach-" * basename(args["nnet"]) * ".png")
savefig(plt, saveto)
println("saved to: $(saveto)")
=#


# dsoln = findCircle(ffnet, x1min, x1max, deepsdp_opts, 2)
# csoln = findCircle(ffnet, x1min, x1max, chordalsdp_opts, 2)


#=
circle2 = findCircle(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes_nosec, solns_nosec = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes0, solns0 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 0)
hplanes2, solns2 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 2)
hplanes4, solns4 = findReach2Dpoly(ffnet, x1min, x1max, chordalsdp_opts, 4)

polys = [hplanes_nosec, hplanes0, hplanes2, hplanes4]

plt = plot()
plt = plotBoundingPolygons!(plt, xfs, polys)
=#


