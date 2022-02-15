module Qc

using LinearAlgebra
using Parameters
using Printf

using ..MyLinearAlgebra
using ..MyNeuralNetwork

# Meta information necessary to construct each Qc
abstract type QcInfo end

# Now that the QCinfo is defined, import the other constructions
include("qc/qc_common.jl")
include("qc/qc_bounded.jl")
include("qc/qc_sector.jl")

export QcInfo, QcBoundedInfo, QcSectorInfo
export vardim, makeQc
export makeA, makeb, makeB

end
