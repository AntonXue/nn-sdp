# Stuff for loading vnnlib specs
EXTS_DIR = joinpath(@__DIR__, "..", "exts")
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))

# Types for safety formulas
const InputSafetyPair = Tuple{QcInputBox, QcSafety}
const DisjClause = Vector{InputSafetyPair}
const CnfSpec = Vector{DisjClause}

#= The read_vnnlib_parser gives us a doubly-nested formula of form.
  NOT φ = OR_{inbox} (OR_{A, b} (x in inbox AND Ay <= b))

That is, the spec file encodes the negation of property φ in disjunctive normal form (DNF).
If NOT φ is satisfiable, then φ is not valid (i.e. the property does not hold).
For us, it is convenient to verify the conjunctive normal form (CNF), so we will convert this to
  φ = AND_{inbox, A, b} OR_{i in 1:length(b)} (x in inbox and Ai y >= bi + ε)
=#
function loadVnnlibCnf(spec_file::String, ffnet::FeedFwdNet; αs=nothing)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]
  # Precalculate the scaling coefficient if one exists
  @assert αs isa Nothing || (αs isa VecReal && length(αs) == ffnet.K)
  α = (αs isa Nothing) ? Nothing : prod(αs)
  # A single term in a conj clause
  cnf_spec = CnfSpec()
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  for input_output in specs
    # In each conj clause the input QC is shared
    inbox, outbox = input_output

    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    qc_input = QcInputBox(x1min=lb, x1max=ub)

    # Each element of outbox is a constraints of form A y <= b,
    # but this is in its negation form, so we flip it to: OR Ai y >= bi + ε
    for out in outbox
      disj_clause = DisjClause()
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)
      for i in 1:length(b)
        println("A[$(i),:]: $(A[i,:]) \t b[$(i)] = $(b[i])")
        ε = 1e-4
        normal = -A[i,:]
        offset = -b[i] - ε
        S = hplaneS(normal, offset, ffnet)
        if αs isa VecReal
          println("SCALING S with α=$(prod(αs)) αs=$(αs)")
          S = scaleS(S, αs, ffnet)
        end
        qc_safety = QcSafety(S=Symmetric(Matrix(S)))
        push!(disj_clause, (qc_input, qc_safety))
      end
      push!(cnf_spec, disj_clause)
    end
  end
  return cnf_spec
end

# Load the queries in CNF form
function loadReluQueriesCnf(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  ffnet, αs = loadFromFileScaled(network_file, NoScaling())
  # ffnet, αs = loadFromFileScaled(network_file, SqrtLogScaling())
  cnf_queries = Vector{Vector{SafetyQuery}}()
  cnf_spec = loadVnnlibCnf(vnnlib_file, ffnet, αs=αs)
  for disj_clause in cnf_spec
    disj_queries = Vector{SafetyQuery}()
    for (qc_input, qc_safety) in disj_clause
      qc_activs = makeQcActivs(ffnet, x1min=qc_input.x1min, x1max=qc_input.x1max, β=β)
      safety_query = SafetyQuery(ffnet=ffnet, qc_input=qc_input, qc_safety=qc_safety, qc_activs=qc_activs)
      push!(disj_queries, safety_query)
    end
    push!(cnf_queries, disj_queries)
  end
  return cnf_queries
end

