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
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 4, # seconds
       "MSK_IPAR_INTPNT_SCALING" => 1,
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-8,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

dopts = DeepSdpOptions(use_dual=true, mosek_opts=mosek_opts, verbose=true)
dndopts = DeepSdpOptions(use_dual=false, mosek_opts=mosek_opts, verbose=true)
copts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:single_decomp)
c2opts = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:double_decomp)

x1min = [2.000; 1.000; -0.174; -1.000]
x1max = [2.200; 1.200; -0.104; -0.800]

makeCartpole(t) = load(joinpath(@__DIR__, "..", "models", "cartpole$(t).pth"))
makeThinCartpole(t) = load(joinpath(@__DIR__, "..", "models", "thin_cartpole$(t).pth"))
ffnet_cartpole = makeCartpole(1)
all_ts = 1:12
all_βs = 0:4

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

# Given a time step and opt, run the specified βs
function go(t, opts, βs; dosave = true, dothin = false)
  if dothin
    saveto = joinpath(DUMP_DIR, "thin_cartpole_t$(t)_$(opts2string(opts)).csv")
  else
    saveto = joinpath(DUMP_DIR, "cartpole_t$(t)_$(opts2string(opts)).csv")
  end
  printstyled("Running $(opts2string(opts)) at t: $(t)\n", color=:green)

  df = DataFrame(beta = Int[],
                 obj_val = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 term_status = String[],
                 eigmax = Real[])

  if dothin
    ffnet = makeThinCartpole(t)
  else
    ffnet = makeCartpole(t)
  end
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
    solve_secs = soln.solve_time
    total_secs = soln.total_time
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
  go(1, dopts, 1:1, dosave=false)
  go(1, copts, 1:1, dosave=false)
  go(1, c2opts, 1:1, dosave=false)
end

# Solve for a particular t wrt all the methods
function runme(t; βs = all_βs)
  go(t, c2opts, βs)
  go(t, copts, βs)
  go(t, dopts, βs)
end

function runtwo(t; βs = all_βs)
  go(t, c2opts, βs)
  go(t, dopts, βs)
end

function runmeThin(t; βs = all_βs)
  go(t, c2opts, βs, dothin=true)
  go(t, copts, βs, dothin=true)
  go(t, dopts, βs, dothin=true)
  go(t, dndopts, βs, dothin=true)
end


printstyled("Warming up!\n", color=:green)
warmup()
printstyled("Warmup done!\n", color=:green)

# runme(1)
# runme(2)
# runme(3)

