# DeepSdp-specific options
@with_kw struct DeepSdpOptions <: QueryOptions
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = false
  verbose::Bool = false
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::DeepSdpOptions)
  setup_start_time = time()
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

  # Now set up the LMI
  Z = Zin + Zout + sum(Zacs)
  @constraint(model, -Z in PSDCone())
  vars[:Z] = Z

  # Compute statistics and return
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Set up a reach query while specifying a generic objective function
function setupReach!(model, query::ReachQuery, opts::DeepSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Make the Zin
  γin = @variable(model, [1:query.qc_input.vardim])
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)
  vars[:γin] = γin

  # And also the Zout and also the objective
  γout = @variable(model, [1:query.qc_reach.vardim])
  @constraint(model, γout[1:query.qc_reach.vardim] .>= 0)
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

  # Calculate stuff and return
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Solve a model that is ready
function solve!(model, vars, opts::DeepSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, summary.solve_time
end

