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

makeCart40(t) = load(joinpath(@__DIR__, "..", "models", "cartw40_step$(t).pth"))
todo_ts = 1:5

function go(todo_qc_activs, opts; dosave = true)
  qc_str = join(string.(todo_qc_activs), "-")
  saveto = joinpath(DUMP_DIR, "cart40_$(qc_str)_$(opts2string(opts)).csv")
  printstyled("running with $(qc_str) | now is: $(now())\n", color=:green)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)

  df = DataFrame(t = Int[],
                 l2gain_squared = Real[],
                 setup_secs = Real[],
                 solve_secs = Real[],
                 total_secs = Real[],
                 term_status = String[],
                 eigmax = Real[])

  for t in todo_ts
    printstyled("\tt: $(t) | now is: $(now())\n", color=:green)
    ffnet = makeCart40(t)
    all_qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

    # Do some filtering on the qc_activs
    qc_activs = Vector{QcActivs}()
    for qca in all_qc_activs
      should_add = (qca isa QcActivBounded && :bounded in todo_qc_activs) ||
                   (qca isa QcActivSector && :sector in todo_qc_activs) ||
                   (qca isa QcActivFinal && :final in todo_qc_activs)
      if should_add
        push!(qc_activs, qca)
      end
    end

    query = ReachQuery(ffnet = ffnet,
                       qc_input = qc_input,
                       qc_activs = qc_activs,
                       qc_reach = QcReachL2Gain(),
                       obj_func = x -> x[1])
    soln = Methods.runQuery(query, opts)
    λmax = eigmax(Symmetric(Matrix(soln.values[:Z])))
    entry = (t,
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

function runme_bounded()
  go([:bounded], dopts, dosave=true)
end

function runme_sector()
  go([:sector], dopts, dosave=true)
end

function runme_final()
  go([:final], dopts, dosave=true)
end

function runme_all()
  go([:bounded, :sector, :final], dopts, dosave=true)
end


