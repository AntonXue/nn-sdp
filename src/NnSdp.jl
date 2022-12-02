module NnSdp

using Dates
using LinearAlgebra

include("MyMath.jl");
include("MyNeuralNetwork/MyNeuralNetwork.jl");
include("Intervals/Intervals.jl");
include("Qc/Qc.jl");
include("Methods/Methods.jl");
include("Utils/Utils.jl");

using Reexport
@reexport using .MyMath
@reexport using .MyNeuralNetwork
@reexport using .Qc
@reexport using .Intervals
@reexport using .Methods
@reexport using .Utils

# Default solver options
DEFAULT_MOSEK_OPTS = Dict()

# The generic formuation that everything ends up calling
function solveQuery(query::Query, opts::QueryOptions)
  println("NnSdp.solveQuery: $(now())")
  soln = Methods.runQuery(query, opts)
  return soln
end

precompile(solveQuery, (Query, DeepSdpOptions))
precompile(solveQuery, (Query, ChordalSdpOptions))

#
function findReachBox(ffnet::FeedFwdNet, x1min::VecReal, x1max::VecReal, β::Int, opts::QueryOptions)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

  outs = Vector{Any}()
  dK1 = ffnet.xdims[end]
  for i in 1:dK1
    eplus, eminus = Vector(e(i, dK1)), Vector(-e(i, dK1))
    push!(outs, (eplus, QcReachHplane(normal=eplus)))
    push!(outs, (eminus, QcReachHplane(normal=eminus)))
    
    # TODO: remove this
    break
  end

  solveds = Vector{Any}()
  for (ei, qc_out) in outs
    println("Trying to do: $(ei)")
    reach_query = ReachQuery(ffnet = ffnet,
                             qc_input = qc_input,
                             qc_activs = qc_activs,
                             qc_reach = qc_out,
                             obj_func = x -> x[1])
    soln = Methods.runQuery(reach_query, opts)
    ρ = soln.values[:γout][1]
    push!(solveds, (ei, ρ, soln.termination_status))
  end
  return solveds
end

#
function findEllipsoid(ffnet::FeedFwdNet, x1min::VecReal, x1max::VecReal, β::Int, opts::QueryOptions)
  # Calculate qc input first
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  # Change the dependency on this one
  qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

  # The output
  P, yc = Utils.approxEllipsoid(ffnet, x1min, x1max)
  invP = Symmetric(inv(P))

  qc_ellipsoid = QcReachEllipsoid(invP=invP, yc=yc)
  reach_query = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = qc_ellipsoid,
                           obj_func = x -> x[1])
  soln = Methods.runQuery(reach_query, opts)

  ρ = soln.values[:γout][1]
  if ρ < 0; ρ = 0 end
  newP = sqrt(ρ) * P
  return newP, yc, soln
end

#
function findCircle(ffnet::FeedFwdNet, x1min::VecReal, x1max::VecReal, β::Int, opts::QueryOptions)
  # Calculate qc input first
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  # Change the dependency on this one
  qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

  # The output
  yc = evalNet(ffnet, (x1max + x1min) / 2)
  qc_circle = QcReachCircle(yc=yc)
  reach_query = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = qc_circle,
                           obj_func = x -> x[1])
  soln = Methods.runQuery(reach_query, opts)
  return soln
end

export solveQuery
export findEllipsoid, findCircle

end

