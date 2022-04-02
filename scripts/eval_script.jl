start_time = time()
include("../src/core/header.jl"); using .Header
include("../src/core/common.jl"); using .Common
include("../src/core/intervals.jl"); using .Intervals
include("../src/core/deep-sdp.jl"); using .DeepSdp
include("../src/core/split-sdp.jl"); using .SplitSdp
include("../src/core/admm-sdp.jl"); using .AdmmSdp
include("../src/parsers/nnet-parser.jl"); using .NNetParser
include("../src/utils.jl"); using .Utils
include("../src/methods.jl"); using .Methods

using LinearAlgebra
using ArgParse
using Printf

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--deepsdp"
      action = :store_true
    "--splitsdp"
      action = :store_true
    "--benchdir"
      help = "the NNet file location"
      arg_type = String
      required = true
    "--tband"
      arg_type = Int
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

# Exactly one must hold
@assert (args["deepsdp"] || args["splitsdp"]) && !(args["deepsdp"] && args["splitsdp"])

# Make sure the relevant directories exist
NNET_DIR = joinpath(args["benchdir"], "nnet")
METHOD_DIR = joinpath(args["benchdir"], (args["deepsdp"] ? "deepsdp" : "splitsdp"))
@assert isdir(args["benchdir"]) && isdir(NNET_DIR)
if !isdir(METHOD_DIR); mkdir(METHOD_DIR) end

# The different batches
DEPTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
BATCH_W5 = ("W5", [(5, d) for d in DEPTHS])
BATCH_W10 = ("W10", [(10, d) for d in DEPTHS])
BATCH_W15 = ("W15", [(15, d) for d in DEPTHS])
BATCH_W20 = ("W20", [(20, d) for d in DEPTHS])
BATCH_TEST = ("TEST", [(5, 5), (5, 10), (10, 5)])

MAX_TIMEOUTS = 2

#
function runSafety(batch, N :: Int; β = nothing)
  batch_id, batch_items = batch
  results = Vector{Any}()

  timeouts_hit = 0
  
  for (ldim, numl) in batch_items
    nnet_filename = @sprintf("rand-in2-out2-ldim%d-numl%d.nnet", ldim, numl)
    nnet_filepath = joinpath(NNET_DIR, nnet_filename)
    @printf("processing NNet: %s\n", nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)

    # Load the appropriate P1 or P2
    if args["deepsdp"]
      tband = args["tband"]
      ffnet, opts = loadP1(nnet_filepath, input, tband=tband)
    else
      tband = args["tband"]
      tband_func = (tband isa Nothing) ? nothing : (x,y) -> tband
      β = min(β, numl - 2)
      ffnet, opts = loadP2(nnet_filepath, input, β, tband_func=tband_func)
    end

    # Safety stuff
    L2gain = sqrt(ldim * numl) # Or something smarter
    # L2gain = 1e6
    @printf("\tL2gain: %.3f\n", L2gain)
    iter_results = Vector{Any}()
    for i in 1:N
      # Do the solve
      soln = solveSafetyL2gain(ffnet, input, opts, L2gain, verbose=true)
      solve_time = soln.solve_time
      term_status = soln.termination_status
      @printf("\t (%d,%d) iter[%d/%d], %s \t solve time: %.3f\n",
              ldim, numl, i, N, term_status, solve_time)
      push!(iter_results, (solve_time, term_status))

      # Break after push so we do store a result
      status_str = string(term_status)
      if status_str == "SLOW_PROGRESS"; break end

      if status_str == "TIME_LIMIT"
        timeouts_hit += 1
        break
      end
    end
    push!(results, (ldim, numl, iter_results))

    # Compute a statistic to print
    avg_solve_time = sum([ir[1] for ir in iter_results]) / (1.0 * N)
    @printf("\t\t\t\t\t   avg time: %.3f\n\n", avg_solve_time)

    if timeouts_hit >= MAX_TIMEOUTS
      break
    end
  end

  # Save the results for this batch
  if args["deepsdp"]
    saveto_filepath = joinpath(METHOD_DIR, @sprintf("P1-%s-runs.txt", batch_id))
  else
    saveto_filepath = joinpath(METHOD_DIR, @sprintf("P2-%s-beta%d-runs.txt", batch_id, β))
  end
  @printf("saving to %s\n", saveto_filepath)
  open(saveto_filepath, "w") do file
    for (ldim, numl, iter_results) in results
      write(file, @sprintf("W %d, D %d\n", ldim, numl))
      for (solve_time, status) in iter_results
        write(file, @sprintf("\tsolve time: %.3f \t %s\n", solve_time, status))
      end
      avg_time = sum(ir[1] for ir in iter_results) / length(iter_results)
      write(file, @sprintf("\tavg time:   %.3f\n", avg_time))
      write(file, "\n")
    end
  end
  return results
end

println("end time: " * string(round(time() - start_time, digits=2)))

# reach_res1 = runReach(1)
# reach_res2 = runReach(2)

warmup(verbose=true)
# safety_res_W10_b1 = runSafety(BATCH_W10, 1, 2)
# safety_res_20 = runSafety(BATCH_W20, 1, 3)



