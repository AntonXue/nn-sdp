# DeepSdp-specific options
@with_kw struct DeepSdpOptions <: QueryOptions
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = false
  verbose::Bool = false
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::DeepSdpOptions)
  vars = Dict()
  # Make the Zin
  γin = @variable(model, [1:query.qc_input.vardim])
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)
  vars[:γin] = γin

  # And the Zout
  Zout = makeZout(query.qc_safety, query.ffnet)

  # Then do the Zacs
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)
  γacs = [vars[Symbol(:γac, i)] for i in 1:length(query.qc_activs)]
  @objective(model, Min, sum(γin) + sum(sum(γac) for γac in γacs))

  # Now set up the LMI
  Z = Zin + Zout + sum(Zacs)
  @constraint(model, -Z in PSDCone())
  vars[:Z] = Z

  return model, vars
end

# Set up a reach query while specifying a generic objective function
function setupReach!(model, query::ReachQuery, opts::DeepSdpOptions)
  vars = Dict()
  # Make the Zin
  γin = @variable(model, [1:query.qc_input.vardim])
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)
  vars[:γin] = γin

  # And also the Zout and also the objective
  γout = @variable(model, [1:query.qc_reach.vardim])
  # @constraint(model, γout[1:query.qc_reach.vardim] .>= 0)
  @objective(model, Min, query.obj_func(γout))
  Zout = makeZout(γout, query.qc_reach, query.ffnet)
  vars[:γout] = γout

  # Then do the Zacs
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Now set up the LMI and objective
  Z = Zin + Zout + sum(Zacs)
  @constraint(model, -Z in PSDCone())
  vars[:Z] = Z

  return model, vars
end

