# Module for different methods
module Methods

using LinearAlgebra
using Parameters 
using Printf
using JuMP
using Dualization
using MosekTools

using ..MyMath
using ..MyNeuralNetwork
using ..Qc

#
DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true)

# Different query specifications
abstract type Query end

# Safety stuff
@with_kw struct SafetyQuery <: Query
  ffnet::FeedFwdNet
  qc_input::QcInput
  qc_safety::QcSafety
  qc_activs::Vector{QcActiv}
  @assert length(qc_activs) >= 1
  qcs::Vector{QcInfo} = [qc_input; qc_safety; qc_activs]
end

# Reachability stuff
@with_kw struct ReachQuery <: Query
  ffnet::FeedFwdNet
  qc_input::QcInput
  qc_reach::QcReach
  qc_activs::Vector{QcActiv}
  @assert length(qc_activs) >= 1
  qcs::Vector{QcInfo} = [qc_input; qc_reach; qc_activs]
  # Vector-input, real-valued objective for the minimization
  obj_func::Function
end

abstract type QueryOptions end

# The solution that is to be output by an algorithm
@with_kw struct QuerySolution{A,B,C,D}
  objective_value::A
  values::B
  summary::C
  termination_status::D
  total_time::Real
  setup_time::Real
  solve_time::Real
end

# Some common functionalities for different methods

# Set up the model
function setupModel!(query::Query, opts::QueryOptions)
  model = opts.use_dual ? Model(dual_optimizer(Mosek.Optimizer)) : Model(Mosek.Optimizer)
  pre_mosek_opts = opts.include_default_mosek_opts ? DEFAULT_MOSEK_OPTS : Dict()
  todo_mosek_opts = merge(pre_mosek_opts, opts.mosek_opts)
  for (k, v) in todo_mosek_opts; set_optimizer_attribute(model, k, v) end
  return model
end

# The Zacs
function setupZacs!(model, query::Query, opts::QueryOptions)
  vars, Zacs = Dict(), Vector{Any}()
  for (i, qc) in enumerate(query.qc_activs)
    γac = @variable(model, [1:qc.vardim])
    vars[Symbol(:γac, i)] = γac
    @constraint(model, γac[1:qc.vardim] .>= 0)
    Zac = makeZac(γac, qc, query.ffnet)
    push!(Zacs, Zac)
  end
  return Zacs, vars
end

# The interface to call; specialize opt-specific stuff as necessary
function runQuery(query::Query, opts::QueryOptions)
  total_start_time = time()
  model = setupModel!(query, opts)

  # Delegate the appropriate call depending on the kind of query
  if query isa SafetyQuery
    _, vars, setup_time = setupSafety!(model, query, opts)
  elseif query isa ReachQuery
    _, vars, setup_time = setupReach!(model, query, opts)
  else
    error("unrecognized query: $(query)")
  end

  # Get ready to return
  summary, values, solve_time = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose;
    @printf("setup: %.3f \tsolve: %.3f \ttotal: %.3f \tvalue: %.4e (%s)\n",
            setup_time, solve_time, total_time,
            objective_value(model), summary.termination_status)
  end
  return QuerySolution(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = solve_time)
end

#
export Query, SafetyQuery, ReachQuery, QueryOptions, QuerySolution

include("deep_sdp.jl")
export DeepSdpOptions, ChordalSdpOptions

include("chordal_sdp.jl")
export ChordalSdpOptions
export DecompMode, OneStage, TwoStage, TwoStageRelaxed

#
export runQuery

end

