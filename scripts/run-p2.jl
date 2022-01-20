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
argparse_settings = ArgParseSettings()
@add_arg_table argparse_settings begin
    "--benchdir"
        help = "the NNet file location"
        arg_type = String
end

args = parse_args(ARGS, argparse_settings)

# Make sure the relevant directories exist
nnet_dir = joinpath(args["benchdir"], "nnet")
p2_dir = joinpath(args["benchdir"], "p2")
@assert isdir(args["benchdir"]) && isdir(nnet_dir)
if !isdir(p2_dir); mkdir(p2_dir) end

# The different batches
DEPTHS = [5, 10, 15, 20, 25, 30]
BATCH_W10 = ("W10", [(10, d) for d in DEPTHS])
BATCH_W20 = ("W20", [(20, d) for d in DEPTHS])
BATCH_W30 = ("W30", [(30, d) for d in DEPTHS])
BATCH_W40 = ("W40", [(40, d) for d in DEPTHS])
BATCH_W40 = ("W50", [(50, d) for d in DEPTHS])

#
function runSafety(batch, β :: Int, N :: Int)
  batch_id, batch_items = batch
  results = Vector{Any}()
  
  for (ldim, numl) in batch_items
    nnet_filename = @sprintf("rand-in2-out2-ldim%d-numl%d.nnet", ldim, numl)
    nnet_filepath = joinpath(nnet_dir, nnet_filename)
    @printf("processing NNet: %s\n", nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)
    β = min(β, numl - 2)
    ffnet, opts = loadP2(nnet_filepath, input, β)

    # Safety stuff
    L2gain = 10.0 # Or something smarter
    iter_results = Vector{Any}()
    for i in 1:N
      soln = solveSafetyL2gain(ffnet, input, opts, L2gain, verbose=true)
      solve_time = soln.solve_time
      term_status = soln.termination_status
      @printf("\t (%d,%d) iter[%d/%d], %s \t solve time: %.3f\n",
              ldim, numl, i, N, term_status, solve_time)
      push!(iter_results, (solve_time, term_status))

      # Break after push so we do store a result
      if string(term_status) == "TIME_LIMIT"; break end
    end
    avg_solve_time = sum([ir[1] for ir in iter_results]) / (1.0 * N)
    @printf("\t\t\t\t\t   avg time: %.3f\n", avg_solve_time)
    @printf("\n")
    push!(results, (ldim, numl, iter_results))
  end

  # Save the results for this batch
  saveto_filepath = joinpath(p2_dir, @sprintf("P2-%s-beta%d-runs.txt", batch_id, β))
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

function runReach(batch, β :: Int)
  results = Vector{Any}()
  for (ldim, numl) in REACH_WIDTH_DEPTHS
    nnet_filename = @sprintf("rand-in2-out2-ldim%d-numl%d.nnet", ldim, numl)
    nnet_filepath = joinpath(nnet_dir, nnet_filename)
    println("processing NNet: " * nnet_filepath)
    @assert isfile(nnet_filepath)

    # Load the thing
    x1min = ones(2) .- 1e-2
    x1max = ones(2) .+ 1e-2
    input = BoxInput(x1min=x1min, x1max=x1max)
    ffnet, opts = loadP2(nnet_filepath, input, β)

    # Safety stuff
    aug_nnet_filename = "β" * string(β) * "_" * nnet_filename
    image_filepath = joinpath(p2_dir, aug_nnet_filename * ".png")
    hplanes, poly_time = solveReachPolytope(ffnet, input, opts, 6, image_filepath)
    xfs = randomTrajectories(10000, ffnet, input.x1min, input.x1max)
    plotReachPolytope(xfs, hplanes, saveto=image_filepath)

    # TODO: write to a text file, probably

    push!(results, (ldim, numl, poly_time))
    println("")
  end
  return results
end

println("end time: " * string(round(time() - start_time, digits=2)))

# reach_res1 = runReach(1)
# reach_res2 = runReach(2)

warmup(verbose=true)
safety_res_10 = runSafety(BATCH_W10, 1, 2)
# safety_res_20 = runSafety(BATCH_W20, 1, 3)



