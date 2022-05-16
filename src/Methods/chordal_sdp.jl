include("chordal_cliques.jl")

# Decomposition modes
abstract type DecompMode end
struct SingleDecomp <: DecompMode end
struct DoubleDecomp <: DecompMode end
struct DoubleRelaxDecomp <: DecompMode end

# Chordal-DeepSdp-specific options
@with_kw struct ChordalSdpOptions <: QueryOptions
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  decomp_mode::DecompMode = SingleDecomp()
  use_dual::Bool = false
  verbose::Bool = false
end

# Set up each Zs involved in the cliques
function setupZs!(model, cliques, query::Query, opts::ChordalSdpOptions)
  Zdim = sum(query.ffnet.zdims)
  Zs, Ecs = Vector{Any}(), Vector{Any}()
  for (Ck, _, Djs) in cliques
    Ckdim = length(Ck)
    # Use two-stage decomposition
    if opts.decomp_mode isa DoubleDecomp || opts.decomp_mode isa DoubleRelaxDecomp
      Ys, Fcs = Vector{Any}(), Vector{Any}()
      for (i, Dj) in enumerate(Djs)
        Djdim = length(Dj)
        Yj = @variable(model, [1:Djdim, 1:Djdim], Symmetric)
        @constraint(model, -Yj in PSDCone())
        push!(Ys, Yj)
        push!(Fcs, Ec(Dj, Ckdim)) # Fcj

        # Break early if we're in relaxed mode, only take the first clique
        if opts.decomp_mode isa DoubleRelaxDecomp && i >= 1; break end
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
  # We do a slice + inject hack for efficiently constructing the Zksum AffExpr
  Zdim = sum(query.ffnet.zdims)
  Zksum = zeros(AffExpr, (Zdim, Zdim))
  cliques = makeCliques(query.qcs, query.ffnet)
  Zs, _ = setupZs!(model, cliques, query, opts)
  @assert length(Zs) == length(cliques)
  # Go through each Zs and manually insert it into Zksum
  for (k, (Ck, Ckparts, _)) in enumerate(cliques)
    Ckdim = length(Ck)
    # For the last clique there is only one part (Zp) so we just add it in
    if k == length(cliques)
      Zksum[(end-Ckdim+1:end), (end-Ckdim+1:end)] += Zs[k]

    # Otherwise we take apart Ckparts
    else
      @assert length(Ckparts) == 2
      Ck1, Ck2 = Ckparts[1], Ckparts[2]
      Ck1dim = length(Ck1)

      # ... and slice the symmetric Zk into four parts
      Zk11 = Zs[k][1:Ck1dim, 1:Ck1dim]
      Zk12 = Zs[k][1:Ck1dim, (Ck1dim+1):end]
      Zk22 = Zs[k][(Ck1dim+1):end, (Ck1dim+1):end]

      # ... and then inject them into Zksum
      Zksum[Ck1, Ck1] += Zk11
      Zksum[Ck1, Ck2] += Zk12
      Zksum[Ck2, Ck1] += Zk12'
      Zksum[Ck2, Ck2] += Zk22
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

  # Set up the big Zksum
  Zksum = setupZksum!(model, query, opts)
  @constraint(model, Z .== Zksum)

  return model, vars
end

# Set up a reach query while specifying a generic objective function
function setupReach!(model, query::ReachQuery, opts::ChordalSdpOptions)
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

  # Set up the big Zksum
  Zksum = setupZksum!(model, query, opts)
  @constraint(model, Z .== Zksum)

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

