# Load Vnnlib
EXTS_DIR = joinpath(@__DIR__, "..", "..", "exts")
include(joinpath(EXTS_DIR, "vnnlib_parser.jl"))

# Types for safety formulas
const InputSafetyPair = Tuple{QcInputBox, QcSafety}
const DisjClause = Vector{InputSafetyPair}
const CnfSpec = Vector{DisjClause}

const ConjClause = Vector{InputSafetyPair}
const DisjSpec = Vector{ConjClause}

#= The read_vnnlib_parser gives us a doubly-nested formula of form.
  NOT φ = OR_{inbox} (OR_{A, b} (x in inbox AND Ay <= b))

That is, the spec file encodes the negation of property φ in disjunctive normal form (DNF).
If NOT φ is satisfiable, then φ is not valid (i.e. the property does not hold).
For us, it is convenient to verify the conjunctive normal form (CNF), so we will convert this to
  φ = AND_{inbox, A, b} OR_{i in 1:length(b)} (x in inbox and Ai y >= bi)
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
        ε = 1e-3
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
  # ffnet, αs = loadFromFileScaled(network_file, SmartScaling())
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
      conj_clause = DisjClause()
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)

      # Go through each row of Ay <= b and that is a conjunction
      for i in 1:length(b)
        println("A[$(i),:]: $(A[i,:]) \t b[$(i)] = $(b[i])")
        S = hplaneS(A[i,:], b[i], ffnet)
        if αs isa VecReal
          println("SCALING S with αs = $(αs)")
          println("α is: $(prod(αs))")
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

# Load the queries in DNF form
function loadReluQueries(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  ffnet, αs = loadFromFileScaled(network_file, NoScaling())
  # ffnet, αs = loadFromFileScaled(network_file, SmartScaling()) # This almost works with 0.5 * sqrt(ck * log(ck))!

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

# Reachability query loading
const SignedQcReach = Tuple{Int, Int, QcReachHplane} # y index, sign, reach
const SignedReachQuery = Tuple{Int, Int, ReachQuery} # y index, sign, reach
const AbSignedReachTuple = Tuple{Matrix, Vector, Vector{SignedReachQuery}}

# Load reach queries based on the spec
function loadVnnlibReach(spec_file::String, ffnet::FeedFwdNet; αs=nothing)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]

  # Precalculate the scaling coefficient if one exists
  @assert αs isa Nothing || (αs isa VecReal && length(αs) == ffnet.K)
  α = (αs isa Nothing) ? Nothing : prod(αs)

  # A single term in a conj clause
  InputReachTuple = Tuple{QcInput, Matrix, Vector, Vector{SignedQcReach}}
  reach_tuples = Vector{InputReachTuple}()
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
      signed_qc_reaches = Vector{SignedQcReach}()

      for yind in 1:outdim
        # If the ith column of A has pos elements, we have pos reach
        if sum(A[:,yind] .> 0) > 0
          normal_pos = VecInt(e(yind, outdim))
          push!(signed_qc_reaches, (yind, 1, QcReachHplane(normal=normal_pos)))
        end

        # If there are negative elements, we have neg reach
        if sum(A[:,yind] .< 0) > 0
          normal_neg = VecInt(-1 * e(yind, outdim))
          push!(signed_qc_reaches, (yind, -1, QcReachHplane(normal=normal_neg)))
        end
      end

      # We need to generate two hplane queries for each hot index in the row
      entry = (qc_input, A, b, signed_qc_reaches)
      push!(reach_tuples, entry)
    end
  end
  return reach_tuples
end

# Load the reachability stuff
function loadReluQueriesReach(network_file::String, vnnlib_file::String, β::Int)
  @assert β >= 0
  reachq_tuples = Vector{AbSignedReachTuple}()
  ffnet, αs = loadFromFileScaled(network_file, NoScaling())
  # ffnet, αs = loadFromFileScaled(network_file, SmartScaling())
  reach_tuples = loadVnnlibReach(vnnlib_file, ffnet, αs=αs)
  for (qc_input, A, b, signed_qc_reaches) in reach_tuples
    signed_reachqs = Vector{SignedReachQuery}()
    qc_activs = makeQcActivs(ffnet, x1min=qc_input.x1min, x1max=qc_input.x1max, β=β)
    for (yind, sgn, qc_reach) in signed_qc_reaches
      obj_func = x -> x[1]
      reach_query = ReachQuery(ffnet=ffnet, qc_input=qc_input, qc_reach=qc_reach, qc_activs=qc_activs, obj_func=obj_func)
      push!(signed_reachqs, (yind, sgn, reach_query))
    end
    push!(reachq_tuples, (A, b, signed_reachqs))
  end
  return reachq_tuples, αs
end



