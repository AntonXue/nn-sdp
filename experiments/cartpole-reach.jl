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

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "cartpole-reach")

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
ts = 1:8

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

# Run a single dim 
function go(dim, opts; only_bounded = true, dosave = true)
  if only_bounded
    saveto = joinpath(DUMP_DIR, "cart40_bounded_dim$(dim)_$(opts2string(opts)).csv")
  else
    saveto = joinpath(DUMP_DIR, "cart40_dim$(dim)_$(opts2string(opts)).csv")
  end
  printstyled("running with dim $(dim) | now is: $(now())\n", color=:green)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  df = DataFrame(t = Int[],
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
  for t in ts
    printstyled("\tdim: $(dim), t: $(t) | now is: $(now())\n", color=:green)
    ffnet = makeCartpole(t)
    all_qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=0)
    qc_activs = Vector{QcActiv}()

    if only_bounded
      for qca in all_qc_activs
        if qca isa QcActivBounded
          push!(qc_activs, qca)
          break
        end
      end
    else
      qc_activs = all_qc_activs
    end

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
    entry = (t,
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

# Run this
function runme()
  go(1, dopts)
  go(2, dopts)
  go(3, dopts)
  go(4, dopts)
end

function runme_bounded()
  go(1, dopts, only_bounded=true)
  go(2, dopts, only_bounded=true)
  go(3, dopts, only_bounded=true)
  go(4, dopts, only_bounded=true)
end


