start_time = time()

using LinearAlgebra
using Distributions
using ArgParse
using Printf
using Dates
using DataFrames
using CSV
using Plots
using Random

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

@printf("load done: %.3f\n", time() - start_time)

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "random-beta")

mosek_opts =
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 3, # seconds
       "MSK_IPAR_INTPNT_SCALING" => 1,
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-8,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6) 
dopts = DeepSdpOptions(use_dual=true, mosek_opts=mosek_opts, verbose=true)
copts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:single_decomp)
c2opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:double_decomp)

x1min = [-1.0; -1.0; -1.0; -1.0]
x1max = [1.0; 1.0; 1.0; 1.0]

Random.seed!(1234)
w = 40
used_ffnet = Utils.randomNetwork([4;w;w;w;w;4], σ=0.5)
βs = 0:10

# Run a single dim
function goHplane(dim, opts; dosave = true)
  saveto = joinpath(DUMP_DIR, "randw$(w)_dim$(dim).csv")
  printstyled("running with dim $(dim) | now is: $(now())\n", color=:green)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)

  ffnet = used_ffnet
  println("the ffnet is:")
  println(ffnet)

  df = DataFrame(beta = Int[],
                 pos_val = Real[],
                 neg_val = Real[],
                 pos_setup_secs = Real[],
                 neg_setup_secs = Real[],
                 pos_solve_secs = Real[],
                 neg_solve_secs = Real[],
                 pos_total_secs = Real[],
                 neg_total_secs = Real[],
                 pos_term_status = String[],
                 neg_term_status = String[],
                 pos_eigmax = Real[],
                 neg_eigmax = Real[])

  for β in βs
    printstyled("\tdim: $(dim), β: $(β) | now is: $(now())\n", color=:green)

    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

    printstyled("\t\tpositive:\n", color=:green)
    query_pos = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = QcReachHplane(normal=Vector(e(dim,4))),
                           obj_func = x -> x[1])
    soln_pos = Methods.runQuery(query_pos, opts)
    λmax_pos = eigmax(Symmetric(Matrix(soln_pos.values[:Z])))

    printstyled("\t\tnegative:\n", color=:green)
    query_neg = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = QcReachHplane(normal=Vector(-e(dim,4))),
                           obj_func = x -> x[1])
    soln_neg = Methods.runQuery(query_neg, opts)
    λmax_neg = eigmax(Symmetric(Matrix(soln_neg.values[:Z])))
    entry = (β,
             soln_pos.objective_value, 
             soln_neg.objective_value,
             soln_pos.setup_time,
             soln_neg.setup_time,
             soln_pos.solve_time,
             soln_neg.solve_time,
             soln_pos.total_time,
             soln_neg.total_time,
             soln_pos.termination_status,
             soln_neg.termination_status,
             λmax_pos,
             λmax_neg)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
      printstyled("updated $(saveto)\n", color=:green)
    end
  end
end

# Do some ellipsoid computations
function goEllipsoid(opts; dosave = true)
  saveto = joinpath(DUMP_DIR, "randw$(w)_ellipsoid.csv")
  printstyled("running ellipsoid | now is: $(now())\n", color=:green)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)

  ffnet = used_ffnet
  width = ffnet.xdims[2]
  println("the ffnet is:")
  println(ffnet)

  df = DataFrame(beta = Int[],
                 width = Int[],
                 yc1 = Real[],
                 yc2 = Real[],
                 yc3 = Real[],
                 yc4 = Real[],
                 P11 = Real[],
                 P12 = Real[],
                 P13 = Real[],
                 P14 = Real[],
                 P22 = Real[],
                 P23 = Real[],
                 P24 = Real[],
                 P33 = Real[],
                 P34 = Real[],
                 P44 = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 eigmax = Real[])

  for β in βs
    printstyled("\tellipsoid, β: $(β) | now is: $(now())\n", color=:green)

    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

    P, yc, soln = findEllipsoid(ffnet, x1min, x1max, β, opts)
    @assert length(yc) == 4
    @assert size(P) == (4,4)
    λmax = eigmax(Symmetric(Matrix(soln.values[:Z])))

    entry = (β,
             width,
             yc[1],
             yc[2],
             yc[3],
             yc[4],
             P[1,1],
             P[1,2],
             P[1,3],
             P[1,4],
             P[2,2],
             P[2,3],
             P[2,4],
             P[3,3],
             P[3,4],
             P[4,4],
             soln.setup_time,
             soln.solve_time,
             soln.total_time,
             λmax)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
      printstyled("updated $(saveto)\n", color=:green)
    end
  end
end

# It doesn't matter which opts we use once β is fixed since we're doing reach
function runme()
  # All four dims
  goHplane(1, dopts)
  goHplane(2, dopts)
  goHplane(3, dopts)
  goHplane(4, dopts)
end


