# Decomposition modes
abstract type ChordalDecompMode end
struct OneStage <: ChordalDecompMode end
struct TwoStage <: ChordalDecompMode end
struct TwoStageRelaxed <: ChordalDecompMode end

# Chordal-DeepSdp-specific options
@with_kw struct ChordalSdpOptions <: QueryOptions
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  decomp_mode::ChordalDecompMode = OneStage()
  use_dual::Bool = false
  verbose::Bool = false
end

# Do the cliques
function setupCliques!(model, cliques, query::Query, opts::ChordalSdpOptions)
  Zdim = sum(query.ffnet.zdims)
  Zs, Ecs = Vector{Any}(), Vector{Any}()
  for (Ck, Djs) in cliques
    Ckdim = length(Ck)
    # Use two-stage decomposition
    if opts.decomp_mode isa TwoStage || opts.decomp_mode isa TwoStageRelaxed
      Ys, Fcs = Vector{Any}(), Vector{Any}()
      for (i, Dj) in enumerate(Djs)
        Djdim = length(Dj)
        Yj = @variable(model, [1:Djdim, 1:Djdim], Symmetric)
        @constraint(model, -Yj in PSDCone())
        push!(Ys, Yj)
        push!(Fcs, Ec(Dj, Ckdim)) # Fcj

        # Break early if we're in relaxed mode, only take the first clique
        if opts.decomp_mode isa TwoStageRelaxed && i >= 1; break end
      end
      Zk = sum(Fcs[j]' * Ys[j] * Fcs[j] for j in 1:length(Ys))
    # In the non-two stage case, the original stuff
    else
      Zk = @variable(model, [1:Ckdim, 1:Ckdim], Symmetric)
      @constraint(model, -Zk in PSDCone())
    end
    push!(Zs, Zk)
    push!(Ecs, Ec(Ck, Zdim)) # Eck
  end
  return Zs, Ecs
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::ChordalSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Make the Zin
  γin = @variable(model, [1:query.qc_input.vardim])
  @constraint(model, γin[1:query.qc_input.vardim] .>= 0)
  Zin = makeZin(γin, query.qc_input, query.ffnet)
  vars[:γin] = γin

  # And the Zout
  Zout = makeZout(query.qc_safety, query.ffnet)

  # Then do the Zacs so we can set up Z
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Big Z matrix
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z

  # Set up cliques
  cliques = makeCliques(query.qcs, query.ffnet)
  Zs, Ecs = setupCliques!(model, cliques, query, opts)

  # The equality constraint
  Zksum = sum(Ecs[k]' * Zs[k] * Ecs[k] for k in 1:length(cliques))
  @constraint(model, Z .== Zksum)

  # Compute statistics and return
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Set up a reach query while specifying a generic objective function
function setupReach!(model, obj_func::Function, query::ReachQuery, opts::ChordalSdpOptions)
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
  @objective(model, Min, obj_func(γout))
  Zout = makeZout(γout, query.qc_reach, query.ffnet)
  vars[:γout] = γout

  # Now the activations
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Big Z matrix
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z

  # Set up cliques
  cliques = makeCliques(query.qcs, query.ffnet)
  Zs, Ecs = setupCliques!(model, cliques, query, opts)

  # The equality constraint
  Zksum = sum(Ecs[k]' * Zs[k] * Ecs[k] for k in 1:length(cliques))
  @constraint(model, Z .== Zksum)

  # Calculate stuff and return
  setup_time = time() - setup_start_time
  return model, vars, setup_time

end

# Solve a model that is ready
function solve!(model, vars, opts::ChordalSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values, summary.solve_time
end

