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

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "cartpole-scale")

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

makeCartpole(t) = load(joinpath(@__DIR__, "..", "models", "cartpole$(t).pth"))
ffnet_cartpole = makeCartpole(1)
all_ts = 1:12
all_βs = 0:6

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

# Given a time step and opt, run the specified βs
function go(t, opts, βs; dosave = true)
  saveto = joinpath(DUMP_DIR, "cartpole_t$(t)_$(opts2string(opts)).csv")
  prinstyled("Running $(opts2string(opts)) at t: $(t)\n", color=:green)

  df = DataFrame(beta = Int[],
                 obj_val = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 term_status = String[],
                 eigmax = Real[])

  ffnet = makeCartpole(t)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)

  for β in βs
    printstyled("\t$(opts2string(opts)) | t: $(t) | β: $(β)\n", color=:green)
    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)
    query = ReachQuery(ffnet = ffnet,
                       qc_input = qc_input,
                       qc_activs = qc_activs,
                       qc_reach = QcReachHplane(normal=Vector(e(1,4))),
                       obj_func = x -> x[1])
    soln = Methods.runQuery(query, opts)
    obj_val = soln.objective_value
    setup_secs = soln.setup_time
    solve_time = soln.solve_time
    total_time = soln.total_time
    term_status = soln.termination_status
    λmax = eigmax(Symmetric(Matrix(soln.values[:Z])))
    entry = (β, obj_val, setup_secs, solve_secs, total_secs, term_status, λmax)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
    end
  end
  return df
end

# A warmup methods
function warmup()
  goOne(1, dopts, 0:1, dosave=false)
  goOne(1, copts, 0:1, dosave=false)
  goOne(1, c2opts, 0:1, dosave=false)
end

# Solve for a particular t wrt all the methods
function runme(t; βs = all_βs)
  goOne(t, dopts, βs)
  goOne(t, copts, βs)
  goOne(t, c2opts, βs)
end

# runme(1)
# runme(2)
# runme(3)

