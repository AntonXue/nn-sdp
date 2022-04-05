module NnSdp

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


end

