# Set up the model
function setupModel!(query::Query, opts::QueryOptions)
  model = opts.use_dual ? Model(dual_optimizer(Mosek.Optimizer)) : Model(Mosek.Optimizer)
  pre_mosek_opts = opts.include_default_mosek_opts ? DEFAULT_MOSEK_OPTS : Dict()
  todo_mosek_opts = merge(pre_mosek_opts, opts.mosek_opts)
  for (k, v) in todo_mosek_opts; set_optimizer_attribute(model, k, v) end
  return model
end

# The Zacs
function setupZacs!(model, query::Query, opts::QueryOptions)
  vars = Dict()
  Zacs = Vector{Any}()
  for (i, qc) in enumerate(query.qc_activs)
    γac = @variable(model, [1:qc.vardim])
    vars[Symbol(:γac, i)] = γac
    @constraint(model, γac[1:qc.vardim] .>= 0)
    Zac = makeZac(γac, qc, query.ffnet)
    push!(Zacs, Zac)
  end
  return Zacs, vars
end

