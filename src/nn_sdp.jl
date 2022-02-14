module NnSdp

include("my_linear_algebra.jl"); using .MyLinearAlgebra
include("my_neural_network.jl"); using .MyNeuralNetwork
include("qc.jl"); using .Qc
# include("intervals.jl")
include("methods.jl"); using .Methods
include("utils.jl"); using .Utils

export MyLinearAlgebra
export MyNeuralNetwork
export Qc
export Methods


# Safety
function solveSafetyL2gain(nnet :: NeuralNetwork, input :: BoxInput, opts, L2gain :: Float64; verbose = false)
  safety = L2gainSafety(L2gain, nnet.xdims)
  safety_prob = SafetyProblem(nnet=nnet, input=input, safety=safety)
  soln = Methods.run(safety_prob, opts)
  return soln
end

# Load a P1
function loadP1(nnet_filepath :: String, input :: BoxInput; verbose = false)
  nnet = NNetParser.NNet(nnet_filepath)

end


end
