module NnSdp


include("my_linear_algebra.jl");
include("my_neural_network.jl");
include("qc.jl");
include("intervals.jl");
include("methods.jl");
include("utils.jl");

using Reexport
@reexport using .MyLinearAlgebra
@reexport using .MyNeuralNetwork
@reexport using .Qc
@reexport using .Intervals
@reexport using .Methods
@reexport using .Utils

# Safety
function solveSafetyL2Gain(nnet :: NeuralNetwork, input :: BoxInput, qcinfos, opts, L2gain :: Float64; verbose = false)
  safety = L2gainSafety(L2gain, nnet.xdims)
  safety_prob = SafetyProblem(nnet=nnet, input=input, output=safety, qcinfos=qcinfos)
  soln = Methods.run(safety_prob, opts)
  return soln
end

# Load a P1
function solveHplaneReach(nnet :: NeuralNetwork, input :: BoxInput, qcinfos, opts, normal :: VecF64; verbose = false)
  hplane = HplaneReachSet(normal=normal)
  reach_prob = ReachProblem(nnet=nnet, input=input, reach=hplane, qcinfos=qcinfos)
  soln = Methods.run(reach_prob, opts)
  return soln
end

export solveSafetyL2Gain, solveHplaneReach

end
