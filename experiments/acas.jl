using LinearAlgebra
using Dates
using NaturalSort
using ArgParse

include("../src/NnSdp.jl"); using .NnSdp

# Where all the files are
ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")
ALL_FILES = readdir(ACAS_DIR, join=true)

ACAS_FILES = filter(f -> match(r".*.onnx", f) isa RegexMatch, ALL_FILES)
ACAS_FILES = sort(ACAS_FILES, lt=natural)

SPEC_FILES = filter(f -> match(r".*.vnnlib", f) isa RegexMatch, ALL_FILES)
SPEC_FILES = sort(SPEC_FILES, lt=natural)

#= Custom MOSEK options we'll use for this experiment.
On Mayur's machine a safe query should take <= 3 minutes with two-stage mode,
=#
ACAS_MOSEK_OPTS = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 5, # Time in seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

# How large are we willing to have λmax(Z) be?
NSD_TOL = 1e-4

# Options to use
chordalsdp_opts = ChordalSdpOptions(mosek_opts=ACAS_MOSEK_OPTS, decomp_mode=TwoStage())

function isSolutionGood(soln::QuerySolution)
  return (soln.termination_status == "OPTIMAL"
          || eigmax(Matrix(soln.values[:Z])) <= NSD_TOL)
end

function extractAcasId(acas_file::String)
end

function extractPropId(prop_file::String)
end



#= Verify an acas network (*.onnx) and property (*.vnnlib) and track:
  * ACAS file used
  * Property tested
  * Number of queries
  * Avg time per successful query
  * Verification result
=#
function verifyAcasProp(acas_file::String, prop_file::String, opts::QueryOptions)
  β = 1 # To be parametrized, maybe
  dnf_queries = loadReluQueries(acas_file, prop_file, β)
  num_queries = length(vcat(dnf_queries...))

  prop_holds = false
  all_solns = Vector{Any}()
  for conj in dnf_queries
    # Property holds when any conjunction is true
    conj_holds = true
    for query in conj
      soln = solveQuery(query, opts)
      soln_good = isSolutionGood(soln)
      conj_holds = conj_holds && soln_good # Update the conj truthiness
      push!(all_solns, soln)

      # If this conj is false, we immediately break to look at the next conj
      if !conj_holds; break end
    end

    if conj_holds
      prop_holds = true
      break
    end
    
  end
  return all_solns, num_queries, prop_holds
end


