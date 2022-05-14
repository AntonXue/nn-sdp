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
ALL_PROP_PAIRS = [PROP1_PAIRS; PROP2_PAIRS; PROP3_PAIRS; PROP4_PAIRS]

# Some test pairs
# TEST_PAIRS = [(ind2acas(i,j), ind2spec(k)) for i in [1] for j in [1] for k in [1]]
TEST_PAIRS = [
              # (ind2acas(1,1), "/home/antonxue/stuff/test/a-b-crown/prop_6_a.vnnlib")
              # (ind2acas(1,1), "/home/antonxue/stuff/test/a-b-crown/prop_6_c.vnnlib")
              # (ind2acas(1,1), ind2spec(1)),
              # (ind2acas(1,2), ind2spec(1)),
              # (ind2acas(1,3), ind2spec(1)),
              # (ind2acas(1,4), ind2spec(1)),
              # (ind2acas(1,5), ind2spec(1)),
              # (ind2acas(1,1), ind2spec(5)),
              # (ind2acas(4,5), ind2spec(10)), # Row 7
              # (ind2acas(1,1), ind2spec(6)),  # Row 2, 3
               # (ind2acas(3,3), ind2spec(9)),  # Row 6
               # (ind2acas(2,9), ind2spec(8)),  # Row 5
              (ind2acas(1,7), ind2spec(3)),
              # (ind2acas(1,8), ind2spec(3)),
              # (ind2acas(1,9), ind2spec(3)),
             ]

#= Custom MOSEK options we'll use for this experiment.
On Mayur's machine a safe query should take <= 3 minutes with two-stage mode,
=#
ACAS_MOSEK_OPTS = 
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 30, # Time in seconds
       # "MSK_IPAR_INTPNT_SCALING" => 1,  # None
       "MSK_IPAR_INTPNT_SCALING" => 2,  # Moderate (preferable, maybe)
       # "MSK_IPAR_INTPNT_SCALING" => 3,  # Aggressive
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
       # "MSK_IPAR_INTPNT_MAX_ITERATIONS" => 500,
       # "MSK_DPAR_DATA_SYM_MAT_TOL" => 1e-10,
       "INTPNT_CO_TOL_REL_GAP" => 1e-9,
       "INTPNT_CO_TOL_PFEAS" => 1e-9,
       "INTPNT_CO_TOL_DFEAS" => 1e-9)

# How large are we willing to have λmax(Z) be?
# NSD_TOL = 1e-4
# NSD_TOL = 5e-3 # This is too high for the small table: everything is safe
# NSD_TOL = 5e-4 # This is still too high: prop 8 network 2-9 is incorrectly marked safe
# NSD_TOL = 1e-4 # Too high for prop 8 network 2-9?
# NSD_TOL = 1e-6
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
function verifyAcasSpec(acas_file::String, spec_file::String, β::Int, opts::QueryOptions)
  cnf_queries = loadReluQueries(acas_file, spec_file, β)
  num_queries = length(vcat(cnf_queries...))

  verif_status = :unknown

  spec_holds = true
  all_solns = Vector{Any}()
  conj_holds = true
  for (disj_ind, disj_clause) in enumerate(cnf_queries)
    println("\tdisj [$(disj_ind)/$(length(cnf_queries))] has $(length(disj_clause)) subqueries | now: $(now())")

    disj_holds = false
    for (query_ind, query) in enumerate(disj_clause)
      #=
      xfs = Utils.sampleTrajs(query.ffnet, query.qc_input.x1min, query.qc_input.x1max)
      ymax = maximum(xfs)
      ymin = minimum(xfs)
      println("\t\tsampled ymax: $(ymax)")
      println("\t\tsampled ymin: $(ymin)")

      # Are there any where y1 => y2?
      y2diffs = [xf[2] - xf[1] for xf in xfs]
      y2negs = sum(y2diffs .< 0)

      # Are there any where y1 => y3?
      y3diffs = [xf[3] - xf[1] for xf in xfs]
      y3negs = sum(y3diffs .< 0)

      # Are there any where y1 => y4?
      y4diffs = [xf[4] - xf[1] for xf in xfs]
      y4negs = sum(y4diffs .< 0)

      # Are there any where y1 => y5?
      y5diffs = [xf[5] - xf[1] for xf in xfs]
      y5negs = sum(y5diffs .< 0)

      println("y2negs: $(y2negs)")
      println("y3negs: $(y3negs)")
      println("y4negs: $(y4negs)")
      println("y5negs: $(y5negs)")
      =#


      println("\t\tsubquery [$(query_ind)/$(length(disj_clause))] of $(basename(acas_file)) | $(basename(spec_file))")
      soln = solveQuery(query, opts)
      is_good = isSolutionGood(soln)
      disj_holds = disj_holds || is_good
      push!(all_solns, soln)

      λmax = eigmax(Matrix(soln.values[:Z]))
      λmin = eigmin(Matrix(soln.values[:Z]))
      println("\t\ttime: $(soln.total_time) | result: $(soln.termination_status) | λs: ($(λmax), $(λmin))")

      # If any query in the disjunctive clause holds, the clause is immediately satisfied
      if disj_holds; break end
    end

    # If this disjunctive clause does not hold, then the overall cnf spec is not valid
    if !disj_holds
      spec_holds = false
      verif_status = :unsafe
      break
    end
  end

  if spec_holds; verif_status = :safe end

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

  qdf = DataFrame(acas=String[], spec=String[], qnum=Int[], num_queries=Int[], time=Float64[], status=String[], eigmax=Float64[])
  qdf_saveto = saveto * "-qdf.csv"

  for (i, (acas_file, spec_file)) in enumerate(pairs)
    println("pair [$(i)/$(length(pairs))] | now: $(now())")
    println("\tacas: $(acas_file)")
    println("\tspec: $(spec_file)")

    all_solns, num_queries, verif_status = verifyAcasSpec(acas_file, spec_file, β, opts)

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

    # Update the qdf
    for (i, soln) in enumerate(all_solns)
      eigmaxZ = eigmax(Symmetric(Matrix(soln.values[:Z])))
      qentry = (acas_name, spec_name, i, num_queries, soln.total_time, soln.termination_status, eigmaxZ)
      push!(qdf, qentry)
    end
    CSV.write(qdf_saveto, qdf)

  end
  return df
end


# Solve a reachability version of the spec
function reachAcasSpec(acas_file::String, spec_file::String, β::Int, opts::QueryOptions)
  reachq_tuples, αs = loadReluQueriesReach(acas_file, spec_file, β)
  α = prod(αs)
  println("α is: $(α)")
  for (A, b, signed_reachqs) in reachq_tuples
    signed_reach_solns = Vector{QuerySolution}()
    for (yind, sgn, reachq) in signed_reachqs
      xfs = Utils.sampleTrajs(reachq.ffnet, reachq.qc_input.x1min, reachq.qc_input.x1max)
      y1min = minimum(xf[1] for xf in xfs)
      y1max = maximum(xf[1] for xf in xfs)
      println("sampled y1min/y1max: ($(y1min), $(y1max))")


      println("y$(yind) has sign $(sgn)")
      soln = NnSdp.solveQuery(reachq, opts)
      obj_val = soln.objective_value
      println("y$(yind) raw obj: $(obj_val), normalized obj: $(obj_val / α)")
      push!(signed_reach_solns, soln)
      println("")
    end
  end

  println("αs: $(αs)")
  println("α: $(α)")
end




# Here begins the stuff that we can customize and try things with

# Options to use
# DOPTS = DeepSdpOptions(verbose=true, mosek_opts=ACAS_MOSEK_OPTS)
DOPTS = DeepSdpOptions(use_dual=true, verbose=true, mosek_opts=ACAS_MOSEK_OPTS)
COPTS = ChordalSdpOptions(verbose=true, mosek_opts=ACAS_MOSEK_OPTS, decomp_mode=TwoStage())
# COPTS = ChordalSdpOptions(use_dual=true, verbose=true, mosek_opts=ACAS_MOSEK_OPTS, decomp_mode=TwoStage())

function gotest(β::Int, opts)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_test.csv")
  df = verifyPairs(TEST_PAIRS, β, opts, saveto)
  println("took time: $(time() - start_time)")
  return df
end

function gotable(β::Int, opts)
  start_time = time()
  saveto = joinpath(DUMP_DIR, "acas_table_beta$(β).csv")
  df = verifyPairs(TABLE_PAIRS, β, opts, saveto)
  println("took time: $(time() - start_time)")
  return df
end


function goprop(k::Int, β::Int, opts=COPTS)
  start_time = time()
  @assert k in [1,2,3,4]
  if k == 1
    prop_pairs = PROP1_PAIRS
  elseif k == 2
    prop_pairs = PROP2_PAIRS
  elseif k == 3
    prop_pairs = PROP3_PAIRS
  elseif k == 4
    prop_pairs = PROP4_PAIRS
  else
    error("not checking prop $(k) right now")
  end
  saveto = joinpath(DUMP_DIR, "acas_prop$(k)_beta$(β).csv")
  df = verifyPairs(prop_pairs, β, opts, saveto)
  println("took time: $(time() - start_time)")
  return df
end



ffnet11 = loadFromFile(ind2acas(1,1))

SIMPLE_PROP1 = "/home/antonxue/stuff/test/a-b-crown/prop_1.vnnlib"
SIMPLE_PROP5 = "/home/antonxue/stuff/test/a-b-crown/prop_5.vnnlib"
SIMPLE_PROP10 = "/home/antonxue/stuff/test/a-b-crown/prop_10.vnnlib"

