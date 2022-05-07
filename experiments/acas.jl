using LinearAlgebra
using Dates
using NaturalSort
using ArgParse
using DataFrames
using CSV

include("../src/NnSdp.jl"); using .NnSdp

# The place where things are
DUMP_DIR = joinpath(@__DIR__, "..", "dump")
ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")

# The ACAS files
ind2acas(i,j) = joinpath(ACAS_DIR, "ACASXU_run2a_$(i)_$(j)_batch_2000.onnx")
ACAS_FILES = [ind2acas(i,j) for i in 1:5 for j in 1:9]
@assert length(ACAS_FILES) == 45

# The spec files
ind2spec(i) = joinpath(ACAS_DIR, "prop_$(i).vnnlib")
SPEC_FILES = [ind2spec(i) for i in 1:10]
@assert length(SPEC_FILES) == 10

# The pairs that we wanna check
TABLE_PAIRS = [(ind2acas(1,1), ind2spec(5)),  # Row 1
               (ind2acas(1,1), ind2spec(6)),  # Row 2, 3
               (ind2acas(1,9), ind2spec(7)),  # Row 4
               (ind2acas(2,9), ind2spec(8)),  # Row 5
               (ind2acas(3,3), ind2spec(9)),  # Row 6
               (ind2acas(4,5), ind2spec(10))] # Row 7

# The different property pairs
PROP1_PAIRS = [(ind2acas(i,j), ind2spec(1)) for i in 1:5 for j in 1:9]
PROP2_PAIRS = [(ind2acas(i,j), ind2spec(2)) for i in 1:5 for j in 1:9]
PROP3_PAIRS = [(ind2acas(i,j), ind2spec(3)) for i in 1:5 for j in 1:9]
PROP4_PAIRS = [(ind2acas(i,j), ind2spec(4)) for i in 1:5 for j in 1:9]

# Some test pairs
TEST_PAIRS = [(ind2acas(i,j), ind2spec(k)) for i in [1] for j in [1] for k in [1]]

#= Custom MOSEK options we'll use for this experiment.
On Mayur's machine a safe query should take <= 3 minutes with two-stage mode,
=#
ACAS_MOSEK_OPTS = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 4, # Time in seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

# How large are we willing to have λmax(Z) be?
NSD_TOL = 1e-4


function isSolutionGood(soln::QuerySolution)
  return (soln.termination_status == "OPTIMAL"
          || eigmax(Matrix(soln.values[:Z])) <= NSD_TOL)
end

#= Verify a single network - spec instance
Goes through each conjunction until one unanimously holds, then returns
* All the query results tried so far
* The total number of queries
* Whether the spec holds
=#
function verifyAcasSpec(acas_file::String, spec_file::String, opts::QueryOptions, β::Int)
  dnf_queries = loadReluQueries(acas_file, spec_file, β)
  num_queries = length(vcat(dnf_queries...))

  verif_status = :unknown

  spec_holds = false
  all_solns = Vector{Any}()
  for (conj_ind, conj) in enumerate(dnf_queries)
    println("\tconjunction [$(conj_ind)/$(length(dnf_queries))] has $(length(conj)) subqueries | now: $(now())")
    # Property holds when any conjunction is true
    conj_holds = true
    for (query_ind, query) in enumerate(conj)
      soln = solveQuery(query, opts)
      is_good = isSolutionGood(soln)
      conj_holds = conj_holds && is_good # Update the conj truthiness
      push!(all_solns, soln)

      λmax = eigmax(Matrix(soln.values[:Z]))

      println("\t\tsubquery [$(query_ind)/$(length(conj))] | time: $(soln.total_time) | result: $(soln.termination_status) | eigmax: $(λmax)")

      # If this conj is false, we immediately break to look at the next conj
      if !conj_holds
        verif_status = :unsafe
        break
      end
    end

    if conj_holds
      verif_status = :safe
      spec_holds = true
      break
    end
  end
  

  verif_total_time = sum([s.total_time for s in all_solns])
  println("")
  println("\tverification status: $(verif_status) | total time: $(verif_total_time)")

  return all_solns, num_queries, verif_status
end

#= Verify an acas network (*.onnx) and spec (*.vnnlib) and track:
  * ACAS file used
  * Property tested
  * Number of queries
  * Avg time per successful query
  * Verification result
=#
function verifyPairs(pairs, β::Int, opts, saveto = joinpath(DUMP_DIR, "hello.csv"))
  num_pairs = length(pairs)
  df = DataFrame(acas=String[], spec=String[], verif_status=String[], num_queries=Int[], queries_ran=Int[], avg_query_time=Float64[], total_time=Float64[])

  for (i, (acas_file, spec_file)) in enumerate(pairs)
    println("pair [$(i)/$(length(pairs))] | now: $(now())")
    println("\tacas: $(acas_file)")
    println("\tspec: $(spec_file)")

    all_solns, num_queries, verif_status = verifyAcasSpec(acas_file, spec_file, opts, β)

    # Get ready to build an entry to the hist, starting with the acas and spec names
    acas_name = basename(acas_file)
    spec_name = basename(spec_file)
    good_solns = filter(isSolutionGood, all_solns)

    if length(good_solns) == 0
      avg_query_time = Inf # infinity time, maybe
    else
      avg_query_time = sum([gs.total_time for gs in good_solns]) / length(good_solns)
    end

    verif_status = String(verif_status)
    queries_ran = length(all_solns)
    total_time = sum([s.total_time for s in all_solns])

    entry = (acas_name, spec_name, verif_status, num_queries, queries_ran, avg_query_time, total_time)
    push!(df, entry)
    println("")

    # For safety, we will re-save the DF every iteration
    CSV.write(saveto, df)
  end
  return df
end

# Here begins the stuff that we can customize and try things with

# Options to use
copts = ChordalSdpOptions(mosek_opts=ACAS_MOSEK_OPTS, decomp_mode=TwoStage())

function gotest(β::Int=2)
  saveto = joinpath(DUMP_DIR, "acas_test.csv")
  verifyPairs(TEST_PAIRS, β, copts, saveto)
end

function go1(β::Int)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_prop1_beta$(β).csv")
  df = verifyPairs(PROP1_PAIRS, β, copts, saveto)
  println("took time: $(time() - start_time)")
  return df
end


function go2(β::Int)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_prop2_beta$(β).csv")
  df = verifyPairs(PROP2_PAIRS, β, copts, saveto)
  println("took time: $(time() - start_time)")
  return df
end


function go3(β::Int)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_prop3_beta$(β).csv")
  df = verifyPairs(PROP3_PAIRS, β, copts, saveto)
  println("took time: $(time() - start_time)")
  return df
end


function go4(β::Int)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_prop4_beta$(β).csv")
  df = verifyPairs(PROP4_PAIRS, β, copts, saveto)
  println("took time: $(time() - start_time)")
  return df
end


