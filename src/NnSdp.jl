module NnSdp

using Dates

include("MyLinearAlgebra.jl");
include("MyNeuralNetwork.jl");
include("Qc/Qc.jl");
include("Intervals/Intervals.jl");
include("Methods/Methods.jl");
include("Utils/Utils.jl");

using Reexport
@reexport using .MyLinearAlgebra
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

# Check whether a network satisfies all the vnnlib stuff
function findReach2Dpoly(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, opts::QueryOptions, β::Int;
                         num_hplanes = 6, use_qc_sector = true)
  # Calculate qc input first
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  qc_activs = Utils.makeQcActivs(ffnet, x1min, x1max, β, use_qc_sector=use_qc_sector)

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

export findReach2Dpoly

end

