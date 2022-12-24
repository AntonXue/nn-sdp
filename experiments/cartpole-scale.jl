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

makeCart40(t) = load(joinpath(@__DIR__, "..", "models", "cartw40_step$(t).pth"))
makeCart10(t) = load(joinpath(@__DIR__, "..", "models", "cartw10_step$(t).pth"))

cart40_ts = 1:25
cart10_ts = 1:7

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

# Run this function with some ts that you wanna do
function go(mode, ts, opts; dosave = true)
  if mode == :cart40
    make_cart_func = makeCart40
    saveto = joinpath(DUMP_DIR, "cart40_$(opts2string(opts)).csv")
  elseif mode == :cart10
    make_cart_func = makeCart10
    saveto = joinpath(DUMP_DIR, "cart10_$(opts2string(opts)).csv")
  else
    error("Unrecognized mode: $(mode)")
  end
  printstyled("Running $(opts2string(opts)) with ts: $(ts)\n", color=:green)

  df = DataFrame(t = Int[],
                 obj_val = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 term_status = String[],
                 eigmax = Real[])
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  for t in ts
    printstyled("\t$(opts2string(opts)) | $(mode) | t: $(t)\n", color=:green)
    ffnet = make_cart_func(t)
    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=0)
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
    entry = (t, obj_val, setup_secs, solve_secs, total_secs, term_status, λmax)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
    end
  end
  return df
end

################

# A warmup methods
function warmup()
  go(:cart10, 2:2, dopts, dosave=false)
  go(:cart10, 2:2, copts, dosave=false)
  go(:cart10, 2:2, c2opts, dosave=false)
end

function runCart10s()
  go(:cart10, cart10_ts, dopts, dosave=true)
  go(:cart10, cart10_ts, c2opts, dosave=true)
  go(:cart10, cart10_ts, copts, dosave=true)
  go(:cart10, cart10_ts, dndopts, dosave=true)
end

function runCart40s()
  go(:cart40, cart40_ts, c2opts, dosave=true)
  go(:cart40, cart40_ts, dopts, dosave=true)
  # go(cart40_ts, dndopts, dosave=true)
end


#################
#
printstyled("Warming up!\n", color=:green)
warmup()
printstyled("Warmup done!\n", color=:green)

# runme(1)
# runme(2)
# runme(3)

