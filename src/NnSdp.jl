module NnSdp

using Dates

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
  soln = Methods.runQuery(query, opts)
  return soln
end

function findCircle(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, opts::QueryOptions, β::Int)
  # Calculate qc input first
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  # Change the dependency on this one
  qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

  # The output
  y0 = evalFeedFwdNet(ffnet, (x1max + x1min) / 2)
  qc_circle = QcReachCircle(y0=y0)

  reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_activs=qc_activs, qc_reach=qc_circle)
  soln = Methods.runQuery(reach_query, opts)
  return soln
end

# Check whether a network satisfies all the vnnlib stuff
function findReach2Dpoly(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, opts::QueryOptions, β::Int;
                         num_hplanes = 6)
  # Calculate qc input first
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  # Change the dependency on this one
  qc_activs = Qc.makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

  hplanes = Vector{Tuple{VecF64, Float64}}()
  solns = Vector{Any}()
  for i in 1:num_hplanes
    θ = ((i-1) / num_hplanes) * 2 * π
    normal = [cos(θ); sin(θ)]
    println("gonna run poly [$(i)/$(num_hplanes)] with normal θ = $(θ); now: $(now())")
    
    qc_reach = QcReachHplane(normal=normal)
    reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_activs=qc_activs, qc_reach=qc_reach)
    soln = Methods.runQuery(reach_query, opts)
    push!(hplanes, (normal, soln.objective_value))
    push!(solns, soln)
  end

  return hplanes, solns
end

export solveQuery
export findCircle, findReach2Dpoly

end

