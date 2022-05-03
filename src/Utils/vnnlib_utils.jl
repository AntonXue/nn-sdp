# Load Vnnlib
EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))

# The Vnnlib formulas are
InputSafety = Tuple{QcInputBox, QcSafety}
InputSafetyScaled = Tuple{QcInputBoxScaled, QcSafety}
const InputSafetyPair = Union{InputSafety, InputSafetyScaled}
const ConjClause = Vector{InputSafetyPair}
const DisjSpec = Vector{ConjClause}

#= Gives a list of input-output safety constraints given the spec file.
VNNLIB uses box inputs, and has output constrs of form Ay <= b

Because DeepSDP can only assert one hyperplane constr at a time,
we break the spec into a disjunctive normal form:
  (P1 AND P2) OR (P3 AND P4) ...
where the Ay <= b generates one query (hyperplane bounds Pi) for each row

Also handle scaling right here
=#
function loadVnnlib(spec_file::String, ffnet::FeedFwdNet; αs=nothing)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]

  # Precalculate the scaling coefficient if one exists
  @assert αs isa Nothing || (αs isa VecF64 && length(αs) == ffnet.K)
  α = (αs isa Nothing) ? Nothing : prod(αs)

  # A single term in a conj clause
  input_safety_dnf = DisjSpec()
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  
  for input_output in specs
    # Build a conjunction clause in each iteration
    conj_clause = ConjClause()
    
    # In each conj clause the input QC is shared
    inbox, outbox = input_output
    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]

    # Use a different QC depending on the scaling
    if αs isa VecF64
      qc_input = QcInputBoxScaled(x1min=lb, x1max=ub, α=α)
    else
      qc_input = QcInputBox(x1min=lb, x1max=ub)
    end

    # Parse out constraints of form A y <= b
    for out in outbox
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)
      for i in 1:length(b)
        S = hplaneS(A[i,:], b[i], ffnet)
        if αs isa VecF64; S = scaleS(S, αs, ffnet) end
        qc_safety = QcSafety(S=S)
        push!(conj_clause, (qc_input, qc_safety))
      end
    end
    push!(input_safety_dnf, conj_clause)
  end
  return input_safety_dnf
end

# Load hte queries in DNF form
function loadReluQueries(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  ffnet, αs = Utils.loadFromFileReluScaled(network_file, 2.0)

  dnf_queries = Vector{Vector{SafetyQuery}}()
  spec = loadVnnlib(vnnlib_file, ffnet, αs=αs)
  for conj in spec
    conj_queries = Vector{SafetyQuery}()
    for (qc_input, qc_safety) in conj
      qc_activs = makeQcActivs(ffnet, qc_input.x1min, qc_input.x1max, β)
      safety_query = SafetyQuery(ffnet=ffnet, qc_input=qc_input, qc_safety=qc_safety, qc_activs=qc_activs)
      push!(conj_queries, safety_query)
    end
    push!(dnf_queries, conj_queries)
  end
  return dnf_queries
end



