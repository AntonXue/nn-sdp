start_time = time()

using LinearAlgebra
using Distributions
using ArgParse
using Printf
using Dates
using DataFrames
using CSV
using Plots

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

@printf("load done: %.3f\n", time() - start_time)

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "cartpole-l2gain") 
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

x1min = [2.000; 1.000; -0.174; -1.000]
x1max = [2.200; 1.200; -0.104; -0.800]

makeCartpole(t) = load(joinpath(@__DIR__, "..", "models", "cartw40_step$(t).pth"))
ffnet_cartpole = makeCartpole(1)
βs = 1:8

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

# Run a single β, dim pair
function go(t, opts; dosave = true)
  saveto = joinpath(DUMP_DIR, "cartw40_step$(t)_$(opts2string(opts)).csv")
  printstyled("running with t: $(t) at dim $(dim) | now is: $(now())\n", color=:green)
  ffnet = makeCartpole(t)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)

  df = DataFrame(beta = Int[],
                 l2gain_squared = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 term_status = String[],
                 eigmax = Real[])

  for β in βs
    printstyled("\tt: $(t), β: $(β) | now is: $(now())\n", color=:green)
    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)
    query = ReachQuery(ffnet = ffnet,
                       qc_input = qc_input,
                       qc_activs = qc_activs,
                       qc_reach = QcReachL2Gain(),
                       obj_func = x -> x[1])
    soln = Methods.runQuery(query, opts)
    λmax = eigmax(Symmetric(Matrix(soln.values[:Z])))
    entry = (β,
             soln.objective_value,
             soln.setup_time,
             soln.solve_time,
             soln.total_time,
             soln.termination_status,
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
  go(1, c2opts)
  go(2, c2opts)
  go(3, c2opts)
  go(4, c2opts)
end
