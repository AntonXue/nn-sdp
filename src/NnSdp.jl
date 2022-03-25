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

# Safety
function solveSafetyL2Gain(nnet::NeuralNetwork, input::BoxInput, qcinfos, opts, L2gain::Float64; verbose = false)
  safety = L2gainSafety(L2gain, nnet.xdims)
  safety_inst = SafetyQuery(nnet=nnet, input=input, output=safety, qcinfos=qcinfos)
  soln = Methods.runQuery(safety_inst, opts)
  return soln
end

# Load a P1
function solveHplaneReach(nnet::NeuralNetwork, input::BoxInput, qcinfos, opts, normal::VecF64; verbose = false)
  hplane = HplaneReachSet(normal=normal)
  reach_inst = ReachQuery(nnet=nnet, input=input, reach=hplane, qcinfos=qcinfos)
  soln = Methods.runQuery(reach_inst, opts)
  return soln
end

export solveSafetyL2Gain, solveHplaneReach

end
