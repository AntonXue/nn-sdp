using Parameters

using ..MyLinearAlgebra
using ..MyNeuralNetwork

@with_kw struct QcSectorInfo <: QcInfo
  dummy :: Bool
end


function makeQc(qcinfo :: QcSectorInfo)
end

