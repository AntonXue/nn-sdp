
# Run interval propagation and quickly generate qc_activs
function makeQcActivs(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, β; use_qc_sector = true)
  intv_info = intervalsWorstCase(x1min, x1max, ffnet)
  println("using worst case intervals")
  # intv_info = intervalsRandomized(x1min, x1max, ffnet)
  # println("using sampled intervals")
  
  # Set up qc bounded
  acdim = sum(ffnet.xdims[2:end-1])
  acymin = vcat([acyi[1] for acyi in intv_info.x_intvs[2:end-1]]...)
  acymax = vcat([acyi[2] for acyi in intv_info.x_intvs[2:end-1]]...)
  qc_bounded = QcActivBounded(acydim=acdim, acymin=acymin, acymax=acymax)
  
  if use_qc_sector
    sec_acxmin = vcat([acxi[1] for acxi in intv_info.acx_intvs]...)
    sec_acxmax = vcat([acxi[2] for acxi in intv_info.acx_intvs]...)
    smin, smax = findSectorMinMax(sec_acxmin, sec_acxmax, ffnet.activ)
    qc_sector = QcActivSector(activ=ReluActiv(), acxdim=acdim, β=β, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

    qc_activs = [qc_bounded; qc_sector]
  else
    qc_activs = [qc_bounded]
  end
  return qc_activs
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



