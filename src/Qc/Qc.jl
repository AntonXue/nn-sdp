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

# Now that the QCinfo is defined, import the other constructions
export QcInfo, QcInput, QcOutput, QcActiv

include("input.jl")
export QcInputBox, QcInputPoly
export makeZin

include("output.jl")
export QcSafety, QcReach
export QcReachHplane, QcReachCircle, QcReachEllipsoid, QcReachL2Gain
export scaleS, makeZout

include("activ.jl")
export QcActivBounded, QcActivSector
export makeSectorMinMax, makeQ, makeA, makeb, makeB
export makeQcActivs, makeZac

end

