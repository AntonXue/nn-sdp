# Load Vnnlib
EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))

# The Vnnlib formulas are
const InputSafetyPair = Tuple{QcInputBox, QcSafety}
const ConjClause = Vector{InputSafetyPair}
const DisjSpec = Vector{ConjClause}

#=
Gives a list of input-output safety constraints given the spec file.
VNNLIB uses box inputs, and has output constrs of form Ay <= b

Because DeepSDP can only assert one hyperplane constr at a time,
we break the spec into a disjunctive normal form:
  (P1 AND P2) OR (P3 AND P4) ...
where the Ay <= b generates one query (hyperplane bounds Pi) for each row
=#
function loadVnnlib(spec_file::String, ffnet::FeedFwdNet)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]

  # A single term in a conj clause
  input_safety_dnf = DisjSpec()
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  
  for input_output in specs
    # Build a conjunction clause in each iteration
    conj_clause = ConjClause()
    
    # In each conj clause the input QC is shared
    inbox, outbox = input_output
    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    qc_input = QcInputBox(x1min=lb, x1max=ub)

    # Parse out constraints of form A y <= b
    for out in outbox
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)
      for i in 1:length(b)
        S = hplaneS(A[i,:], b[i], ffnet)
        qc_safety = QcSafety(S=S)
        push!(conj_clause, (qc_input, qc_safety))
      end
    end
    push!(input_safety_dnf, conj_clause)
  end
  return input_safety_dnf
end

# TODO: make this accept DNFs
function loadQueries(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  ffnet = loadFromFile(network_file)
  spec = loadVnnlib(vnnlib_file, ffnet)
  queries = Vector{SafetyQuery}()
  for (qc_input, qc_safety) in spec
    qc_activs = makeQcActivs(ffnet, qc_input.x1min, qc_input.x1max, β)
    safety_query = SafetyQuery(ffnet=ffnet, qc_input=qc_input, qc_safety=qc_safety, qc_activs=qc_activs)
    push!(queries, safety_query)
  end
  return queries
end



