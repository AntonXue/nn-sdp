module Qc

using LinearAlgebra
using SparseArrays
using Parameters

using ..MyMath
using ..MyNeuralNetwork
using ..Intervals

# Meta information necessary to construct each Qc
abstract type QcInfo end
abstract type QcInput <: QcInfo end
abstract type QcActiv <: QcInfo end
abstract type QcOutput <: QcInfo end

const Clique = VecInt

# Now that the QCinfo is defined, import the other constructions
include("qc_input.jl")
include("qc_output.jl")
include("qc_activ.jl")
include("cliques.jl")

export Clique
export QcInfo, QcInput, QcOutput, QcActiv
export QcInputBox, QcInputPoly
export QcSafety, QcReach, QcReachHplane
export QcActivBounded, QcActivSector
export findSectorMinMax, makeQac, makeA, makeb, makeB
export makeQcActivs
export makeZin, makeZout, makeZac
export findCliques

end

