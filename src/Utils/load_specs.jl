
# Gives a list of input-output safety constraints given the spec file
function loadVnnlib(spec_file::String, ffnet::FeedFwdNet)
  indim, outdim = ffnet.xdims[1], ffnet.xdims[end]
  specs = read_vnnlib_simple(spec_file, indim, outdim)
  input_safety_constrs = Vector{Tuple{InputConstraint, SafetyConstraint}}()
  for spec in specs
    inbox, outbox = spec
    lb, ub = [b[1] for b in inbox], [b[2] for b in inbox]
    input = BoxInput(x1min=lb, x1max=ub)
    # Parse out constraints of form A y <= b
    for out in outbox
      A = hcat(out[1]...)'
      b = out[2]
      @assert size(A)[1] == length(b)

      for i in 1:length(b)
        S = Methods.makeShplane(A[i,:], b[i], ffnet) # TODO: last argument
        safety = SafetyConstraint(S)
        push!(input_safety_constrs, (input, safety))
      end
    end
  end
  return input_safety_constrs
end

