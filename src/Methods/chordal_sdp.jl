# Decomposition modes
abstract type DecompMode end
struct OneStage <: DecompMode end
struct TwoStage <: DecompMode end
struct TwoStageRelaxed <: DecompMode end

# Chordal-DeepSdp-specific options
@with_kw struct ChordalSdpOptions <: QueryOptions
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  decomp_mode::DecompMode = OneStage()
  use_dual::Bool = false
  verbose::Bool = false
end

# Do the cliques
function setupZs!(model, cliques, query::Query, opts::ChordalSdpOptions)
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

# More efficient construction of Zksum
function setupZksum!(model, query::Query, opts::ChordalSdpOptions)
  ffnet = query.ffnet
  zdims, K = query.ffnet.zdims, query.ffnet.K
  Zdim = sum(zdims)
  S(k) = (k == 0) ? 0 : sum(zdims[1:k])
  Zksum = zeros(AffExpr, (Zdim, Zdim))
  cliques = makeCliques(query.qcs, query.ffnet)
  qc_sector = (filter(qc -> qc isa QcActivSector, query.qc_activs))[1]
  β = qc_sector.β
  Zs, Ecs = setupZs!(model, cliques, query, opts)
  for (k, (Ck, _)) in enumerate(cliques)
    Ckdim = length(Ck)
    # For the last clique we just add it in
    if k == length(cliques)
      Zksum[end-Ckdim+1:end, end-Ckdim+1:end] += Zs[k]

    # Otherwise we take apart Zk at specific index ranges to avoid doing multiplication
    else
      Zk11dim = Ckdim - zdims[K] - 1
      Zk11 = Zs[k][1:Zk11dim, 1:Zk11dim]
      Zk12 = Zs[k][1:Zk11dim, (Zk11dim+1):end]
      Zk22 = Zs[k][(Zk11dim+1):end, (Zk11dim+1):end]

      Ck_init1 = S(k-1) + 1
      Ck_initdim = ffnet.zdims[k] + ffnet.zdims[k+1] + β
      Ck_init = Ck_init1 : (Ck_init1 + Ck_initdim - 1)
      Ck_tail1 = S(ffnet.K-1) + 1
      Ck_taildim = ffnet.zdims[ffnet.K] + 1
      Ck_tail = Ck_tail1 : (Ck_tail1 + Ck_taildim - 1)

      Zksum[Ck_init, Ck_init] += Zk11
      Zksum[Ck_init, Ck_tail] += Zk12
      Zksum[Ck_tail, Ck_init] += Zk12'
      Zksum[Ck_tail, Ck_tail] += Zk22
    end
  end
  return Zksum
end

# Set up the model for safety verification (satisfiability)
function setupSafety!(model, query::SafetyQuery, opts::ChordalSdpOptions)
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
  γacs = [vars[Symbol(:γac, i)] for i in 1:length(query.qc_activs)]
  @objective(model, Min, sum(γin) + sum(sum(γac) for γac in γacs))

  # Big Z matrix
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z

  #####################
  # Set up cliques
  cliques = makeCliques(query.qcs, query.ffnet)
  Zs, Ecs = setupZs!(model, cliques, query, opts)

  # The equality constraint
  Zksum = sum(Ecs[k]' * Zs[k] * Ecs[k] for k in 1:length(cliques))
  @constraint(model, Z .== Zksum)

  return model, vars
end

# Set up a reach query while specifying a generic objective function
function setupReach!(model, query::ReachQuery, opts::ChordalSdpOptions)
  init_time = time()

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

  # Now the activations
  Zacs, Zacvars = setupZacs!(model, query, opts)
  vars = merge(vars, Zacvars)

  # Big Z matrix
  Z = Zin + Zout + sum(Zacs)
  vars[:Z] = Z
  printstyled("tick A took time: $(time() - init_time)\n", color=:blue)

  # Set up cliques
  #=
  cliques = makeCliques(query.qcs, query.ffnet)
  Zs, Ecs = setupZs!(model, cliques, query, opts)
  printstyled("tick B took time: $(time() - init_time)\n", color=:blue)

  #################
  # The equality constraint
  Zksum = zeros(size(Z))
  for k in 1:length(cliques)
    # println("test: $(size(Ecs[k]' * Zs[k] * Ecs[k]))")
    Zksum += Ecs[k]' * Zs[k] * Ecs[k]
  end
  # Zksum = sum(Ecs[k]' * Zs[k] * Ecs[k] for k in 1:length(cliques))
  =#

  Zksum = setupZksum!(model, query, opts)

  ################
  printstyled("tick C took time: $(time() - init_time)\n", color=:blue)
  @constraint(model, Z .== Zksum)
  printstyled("tick D took time: $(time() - init_time)\n", color=:blue)

  return model, vars
end

# Solve a model that is ready
function solve!(model, vars, opts::ChordalSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values
end

