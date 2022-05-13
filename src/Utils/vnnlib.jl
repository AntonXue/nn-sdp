# Load Vnnlib
EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))

# The Vnnlib formulas are
const InputSafetyPair = Tuple{QcInputBox, QcSafety}
const ConjClause = Vector{InputSafetyPair}
const DisjSpec = Vector{ConjClause}

const InputAbReaches = Tuple{QcInput, Matrix, Vector, Vector{QcReach}}
const AbReachQueries = Tuple{Matrix, Vector, Vector{ReachQuery}}

#= The read_vnnlib_simple parser gives us a doubly-nested formula of form:

  OR_{inbox} (OR_{A, b} (x in inbox AND Ay <= b))

which we can flatten into a disjunctive normal form (dnf) of
  
  OR_{inbox, A, b} (x in inbox AND 

Because DeepSDP can only assert one hyperplane constr at a time,
we break the spec into a disjunctive normal form:
  (P1 AND P2) OR (P3 AND P4) ...
where the Ay <= b generates one query (hyperplane bounds Pi) for each row

Also handle scaling right here
=#
function loadVnnlib(spec_file::String, ffnet::FeedFwdNet; αs=nothing)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]

  # Precalculate the scaling coefficient if one exists
  @assert αs isa Nothing || (αs isa VecReal && length(αs) == ffnet.K)
  α = (αs isa Nothing) ? Nothing : prod(αs)

  # A single term in a conj clause
  input_safety_dnf = DisjSpec()
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  
  for input_output in specs
    # In each conj clause the input QC is shared
    inbox, outbox = input_output

    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    qc_input = QcInputBox(x1min=lb, x1max=ub)

    # Each element of outbox is a constraints of form A y <= b
    for out in outbox
      conj_clause = ConjClause()
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)

      # Go through each row of Ay <= b and that is a conjunction
      for i in 1:length(b)
        S = hplaneS(A[i,:], b[i], ffnet)
        if αs isa VecReal
          println("SCALING S with αs = $(αs)")
          S = scaleS(S, αs, ffnet)
        end
        qc_safety = QcSafety(S=Symmetric(Matrix(S)))
        push!(conj_clause, (qc_input, qc_safety))
      end
      push!(input_safety_dnf, conj_clause)
    end
  end
  return input_safety_dnf
end

# Load reach queries based on the spec
function loadVnnlibReach(spec_file::String, ffnet::FeedFwdNet; αs=nothing)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]

  # Precalculate the scaling coefficient if one exists
  @assert αs isa Nothing || (αs isa VecReal && length(αs) == ffnet.K)
  α = (αs isa Nothing) ? Nothing : prod(αs)

  # A single term in a conj clause
  input_Ab_reaches_dnf = Vector{InputAbReaches}()
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  
  for input_output in specs
    # In each conj clause the input QC is shared
    inbox, outbox = input_output

    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    qc_input = QcInputBox(x1min=lb, x1max=ub)

    # Each element of outbox is a constraints of form A y <= b
    for out in outbox
      A = hcat(out[1]...)'
      b = out[2]
      qc_reaches = Vector{QcReachHplane}()
      # Find out which outputs we actually care about
      rowhots = ones(size(A)[1])' * abs.(A) .> 0
      println("row hots: $(rowhots)")
      for (i, ishot) in enumerate(rowhots)
        if ishot == 1
          normal_top = zeros(outdim)
          normal_top[i] = 1
          qc_reach_top = QcReachHplane(normal=normal_top)

          normal_bot = zeros(outdim)
          normal_bot[i] = -1
          qc_reach_bot = QcReachHplane(normal=normal_bot)

          push!(qc_reaches, qc_reach_top)
          push!(qc_reaches, qc_reach_bot)

          println("normal_top: $(normal_top)")
          println("normal_bot: $(normal_bot)")
          println("")
        end
      end

      # We need to generate two hplane queries for each hot index in the row
      entry = (qc_input, A, b, qc_reaches)
      push!(input_Ab_reaches_dnf, entry)
    end
  end
  return input_Ab_reaches_dnf
end



# Load hte queries in DNF form
function loadReluQueries(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0

  # ffnet, αs = Utils.loadFromFileReluScaled(network_file)
  # ffnet, αs = MyNeuralNetwork.loadFromFileReluScaledStupid(network_file, 0.99) # eigmax 0.0001
  # ffnet, αs = MyNeuralNetwork.loadFromFileReluFixedWknorm(network_file, 2) # Infeasible, eigmax 7.98
  # ffnet, αs = MyNeuralNetwork.loadFromFileReluFixedWknorm(network_file, 4) # Slow progress, eigmax 0.798
  # ffnet, αs = MyNeuralNetwork.loadFromFileReluFixedWknorm(network_file, 4)

  # ffnet = Utils.loadFromFile(network_file)
  # αs = ones(ffnet.K)

  # ffnet, αs = loadFromFileScaled(network_file, FixedConstScaling(0.99))
  # ffnet, αs = loadFromFileScaled(network_file, NoScaling())
  ffnet, αs = loadFromFileScaled(network_file, SmartScaling()) # This almost works with 0.5 * sqrt(ck * log(ck))!

  dnf_queries = Vector{Vector{SafetyQuery}}()
  spec = loadVnnlib(vnnlib_file, ffnet, αs=αs)
  for conj in spec
    conj_queries = Vector{SafetyQuery}()
    for (qc_input, qc_safety) in conj
      qc_activs = makeQcActivs(ffnet, x1min=qc_input.x1min, x1max=qc_input.x1max, β=β)
      safety_query = SafetyQuery(ffnet=ffnet, qc_input=qc_input, qc_safety=qc_safety, qc_activs=qc_activs)
      push!(conj_queries, safety_query)
    end
    push!(dnf_queries, conj_queries)
  end
  return dnf_queries
end

function loadReluQueriesReach(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  dnf_Ab_reachqs = Vector{AbReachQueries}()

  ffnet, αs = loadFromFileScaled(network_file, NoScaling())
  input_Ab_reaches_dnf = loadVnnlibReach(vnnlib_file, ffnet, αs=αs)
  for (qc_input, A, b, qc_reaches) in input_Ab_reaches_dnf
    reachqs = Vector{ReachQuery}()
    qc_activs = makeQcActivs(ffnet, x1min=qc_input.x1min, x1max=qc_input.x1max, β=β)
    for qc_reach in qc_reaches
      obj_func = x -> x[1]
      reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_reach=qc_reach, qc_activs=qc_activs, obj_func=obj_func)
      push!(reachqs, reach_query)
    end
    push!(dnf_Ab_reachqs, (A, b, reachqs))
  end
  return dnf_Ab_reachqs
end




