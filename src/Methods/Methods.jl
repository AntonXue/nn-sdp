module Methods

using LinearAlgebra
using Parameters 
using Printf
using JuMP
using Dualization
using MosekTools
using Dates

using ..MyLinearAlgebra
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
end

abstract type QueryOptions end

# The solution that is to be output by an algorithm
@with_kw struct QuerySolution{A,B,C,D}
  objective_value::A
  values::B
  summary::C
  termination_status::D
  total_time::Float64
  setup_time::Float64
  solve_time::Float64
end

include("methods_common.jl")
include("deep_sdp.jl")
include("chordal_sdp.jl")

export Query, SafetyQuery, ReachQuery, QueryOptions, QuerySolution
export DeepSdpOptions, ChordalSdpOptions
export runQuery

end

