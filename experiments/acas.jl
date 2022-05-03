using LinearAlgebra
using Dates
using ArgParse


ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")
ALL_FILES = readdir(ACAS_DIR, join=true)
ACAS_FILES = filter(f -> match(r".*.onnx", f) isa RegexMatch, ALL_FILES)
SPEC_FILES = filter(f -> match(r".*.vnnlib", f) isa RegexMatch, ALL_FILES)


function verify_acas_prop(acas_file::String, prop_file:String)

end


