
# Options
@with_kw struct DeepSdpOptions <: QueryOptions
  max_solve_time::Float64 = 60.0 * 20 # seconds
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = false
  verbose::Bool = false
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::DeepSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Make the Zin and Zout first
  γin = @variable(model, [1:query.qc_input.vardim])
  vars[:γin] = γin
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)
  Zout = makeZout(query.qc_safety, query.ffnet)

  # Then do the Zacs
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Now set up the LMI
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z
  @constraint(model, -Z in PSDCone())

  # Compute statistics and return
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Hyperplane reachability setup
function setupHplaneReach!(model, query::ReachQuery, opts::DeepSdpOptions)
  @assert query.qc_reach isa QcReachHplane
  setup_start_time = time()
  vars = Dict()

  # Make the Zin
  γin = @variable(model, [1:query.qc_input.vardim])
  vars[:γin] = γin
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)

  # And also the Zout and also the objective
  γout = @variable(model)
  vars[:γout] = γout
  @constraint(model, γout >= 0)
  @objective(model, Min, γout)
  Zout = makeZout([γout], query.qc_reach, query.ffnet)

  # Then do the Zacs
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Now set up the LMI and objective
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z
  @constraint(model, -Z in PSDCone())

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

# The interface to call
function runQuery(query::Query, opts::DeepSdpOptions)
  total_start_time = time()
  model = setupModel!(query, opts)

  # Delegate the appropriate call depending on the kind of query
  if query isa SafetyQuery
    _, vars, setup_time = setupSafety!(model, query, opts)
  elseif query isa ReachQuery && query.qc_reach isa QcReachHplane
    _, vars, setup_time = setupHplaneReach!(model, query, opts)
  else
    error("\tunrecognized query: $(query)")
  end

  # Get ready to return
  if opts.verbose; println("\tsetup done at: $(now())") end

  summary, values, solve_time = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose;
    @printf("\tsetup: %.3f \tsolve: %.3f \ttotal: %.3f \tvalue: %.4e (%s)\n",
            setup_time, solve_time, total_time,
            objective_value(model), summary.termination_status)
  end
  return QuerySolution(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = solve_time)
end

