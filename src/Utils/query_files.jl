
# Gives a list of input-output safety constraints given the spec file
# Vnnlib uses box inputs
function loadVnnlib(spec_file::String, ffnet::FeedFwdNet)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  qcs_input_safety = Vector{Tuple{QcInputBox, QcSafety}}()

  for spec in specs
    inbox, outbox = spec
    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    qc_input = QcInputBox(x1min=lb, x1max=ub)

    # Parse out constraints of form A y <= b
    for out in outbox
      A = hcat(out[1]...)'
      b = out[2]
      # println("making specs relaxed")
      # b = b .+ 1
      @assert size(A)[1] == length(b)
      for i in 1:length(b)
        S = hplaneS(A[i,:], b[i], ffnet)
        qc_safety = QcSafety(S=S)
        push!(qcs_input_safety, (qc_input, qc_safety))
      end
    end
  end
  return qcs_input_safety
end

function loadQueries(network_file, vnnlib_file, β)
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



