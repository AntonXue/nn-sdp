using LinearAlgebra
using Dates
using NaturalSort
using ArgParse
using DataFrames
using CSV

include("../src/NnSdp.jl"); using .NnSdp

# The place where things are
DUMP_DIR = joinpath(@__DIR__, "..", "dump")
RAND_DIR = joinpath(@__DIR__, "..", "bench", "rand")


# Make a query depending on the type of method
function makeOurStyleQuery(ffnet::FeedFwdNet, x1min::Vector, x1max::Vector, β::Int)
  qc_input = QcInput(x1min=x1min, x1max=x1max)
  qc_activs = makeQcActivs(ffnet, x1min, x1max, β)
  outdim = ffnet.xdims[end]
  qc_reach = QcReachHplane(normal=e(1,outdim))
  obj_func = x -> x[1]
  query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_reach=qc_reach, qc_activs=qc_activs, obj_func=obj_func)
  return query
end

# Call this
function makeQuery(ffnet::FeedFwdNet, x1min::Vector, x1max::Vector, method::Symbol; β=nothing)
  if method == :deepsdp || method == :chordalsdp
    return makeOurStyleQuery(ffnet, x1min, x1max, β)
  else
    error("unrecognized method: $(method)")
  end
end




