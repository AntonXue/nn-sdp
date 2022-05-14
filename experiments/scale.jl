script_start_time = time()
using LinearAlgebra
using Dates
using NaturalSort
using ArgParse
using DataFrames
using CSV

include("../src/NnSdp.jl"); using .NnSdp

# The place where things are
DUMP_DIR = joinpath(@__DIR__, "..", "dump", "scale")
RAND_DIR = joinpath(@__DIR__, "..", "bench", "rand")
dims2rand(width, depth) = joinpath(RAND_DIR, "rand-I2-O2-W$(width)-D$(depth).nnet")

DEPTHS = 5:5:50
SMALL_FILES = [dims2rand(10, d) for d in [5, 10]]
W10_FILES = [dims2rand(10, d) for d in DEPTHS]
W20_FILES = [dims2rand(20, d) for d in DEPTHS]
W30_FILES = [dims2rand(30, d) for d in DEPTHS]
W40_FILES = [dims2rand(40, d) for d in DEPTHS]


X1MIN = 1e2 * ones(2) .- 5e-1
X1MAX = 1e2 * ones(2) .+ 5e-1
BETAS = [0, 1, 2, 3, 4, 5]
NSD_TOL = 5e-4

SCALE_MOSEK_OPTS = 
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 2, # Time in seconds
       # "MSK_IPAR_INTPNT_SCALING" => 1,  # None
       "MSK_IPAR_INTPNT_SCALING" => 2,  # Moderate (preferable, maybe)
       # "MSK_IPAR_INTPNT_SCALING" => 3,  # Aggressive
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
       # "MSK_IPAR_INTPNT_MAX_ITERATIONS" => 500,
       # "MSK_DPAR_DATA_SYM_MAT_TOL" => 1e-10,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

DOPTS = DeepSdpOptions(use_dual=true, verbose=true, mosek_opts=SCALE_MOSEK_OPTS)
COPTS = ChordalSdpOptions(verbose=true, mosek_opts=SCALE_MOSEK_OPTS, decomp_mode=OneStage())
C2OPTS = ChordalSdpOptions(verbose=true, mosek_opts=SCALE_MOSEK_OPTS, decomp_mode=TwoStage())

# Call this to make a query
function makeQuery(ffnet::FeedFwdNet, x1min::Vector, x1max::Vector, method::Symbol; β=nothing)
  # If we're using one of our methods, the query construction is consistent
  if method == :deepsdp || method == :chordalsdp || method == :chordalsdp2
    qc_input = QcInputBox(x1min=x1min, x1max=x1max)
    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)
    normal = (1e2 * 2^log(ffnet.K)) * Vector{Float64}(e(1, ffnet.xdims[end]))
    qc_reach = QcReachHplane(normal=normal)
    obj_func = x -> x[1]
    query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_reach=qc_reach, qc_activs=qc_activs, obj_func=obj_func)
  # For other ones we need to special case
  else
    error("unrecognized method: $(method)")
  end
  return query
end

# Run a particular width-depth configuration
function runFileOurStyle(network_file::String, method::Symbol; dosave::Bool=true)
  # Figure out the queries we need to run
  if method == :deepsdp; opts = DOPTS
  elseif method == :chordalsdp; opts = COPTS
  elseif method == :chordalsdp2; opts = C2OPTS
  else; error("unrecognized method: $(method)")
  end

  ffnet, αs = loadFromFileScaled(network_file, SqrtLogScaling())
  β_query_pairs = [(β, makeQuery(ffnet, X1MIN, X1MAX, method, β=β)) for β in BETAS]

  # Now run each query
  saveto = joinpath(DUMP_DIR, "$(string(method))-$(basename(network_file)).csv")
  df = DataFrame(beta=Int[], total_secs=Real[], obj_val=Real[], term_status=String[], eigmax=Real[])
  for (β, query) in β_query_pairs
    println("Running $(basename(network_file)) | $(method) | β: $(β)")
    soln = NnSdp.solveQuery(query, opts)
    total_secs = soln.total_time
    obj_val = soln.objective_value
    status = soln.termination_status
    λmax = eigmax(Matrix(soln.values[:Z]))
    entry = (β, total_secs, obj_val, status, λmax)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
      # printstyled("updated $(saveto)\n", color=:green)
    end
  end
  return df
end

# Call this
function runFile(network_file::String, method::Symbol; dosave::Bool=true)
  if method == :deepsdp || method == :chordalsdp || method == :chordalsdp2
    df = runFileOurStyle(network_file, method, dosave=dosave)
  else
    error("unrecognized method: $(method)")
  end
  return df
end

# Run a bunch of files
function runFileBatch(network_files::Vector{String}, method::Symbol)
  res = Vector{Any}()
  for file in network_files
    printstyled("file: $(file)\n", color=:green)
    start_time = time()
    df = runFile(file, method)
    run_time = time() - start_time
    entry = (file, df, run_time)
    println("\n")
  end
  return res
end

function warmup()
  runFile(dims2rand(5,5), :deepsdp, dosave=false)
  # runFile(dims2rand(5,5), :chordalsdp, dosave=false)
  runFile(dims2rand(5,5), :chordalsdp2, dosave=false)
end


# Stuff happens here
printstyled("starting warmup\n", color=:green)
warmup()
printstyled("script start to here took time: $(time() - script_start_time)\n\n\n", color=:green)


#=
c_small_res = runFileBatch(SMALL_FILES, :chordalsdp)
c2_small_res = runFileBatch(SMALL_FILES, :chordalsdp2)
d_small_res = runFileBatch(SMALL_FILES, :deepsdp)
=#

# W10
c_W10_res = runFileBatch(W10_FILES, :chordalsdp)
c2_W10_res = runFileBatch(W10_FILES, :chordalsdp2)
d_W10_res = runFileBatch(W10_FILES, :deepsdp)


# W20
c_W20_res = runFileBatch(W20_FILES, :chordalsdp)
c2_W20_res = runFileBatch(W20_FILES, :chordalsdp2)
d_W20_res = runFileBatch(W20_FILES, :deepsdp)


# W30
c_W30_res = runFileBatch(W30_FILES, :chordalsdp)
c2_W30_res = runFileBatch(W30_FILES, :chordalsdp2)
d_W30_res = runFileBatch(W30_FILES, :deepsdp)



