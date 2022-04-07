
# Run interval propagation and quickly generate qc_activs
function makeQcActivs(ffnet::FeedFwdNet, x1min::VecF64, x1max::VecF64, β)
  intv_info = intervalsWorstCase(x1min, x1max, ffnet)
  # intv_info = intervalsRandomized(x1min, x1max, ffnet)
  
  # Set up qc bounded
  acxdim = sum(ffnet.xdims[2:end-1])
  acxmin = vcat([acxi[1] for acxi in intv_info.x_intvs[2:end-1]]...)
  acxmax = vcat([acxi[2] for acxi in intv_info.x_intvs[2:end-1]]...)
  qc_bounded = QcActivBounded(acxdim=acxdim, acxmin=acxmin, acxmax=acxmax)
  
  sec_acxmin = vcat([acxi[1] for acxi in intv_info.acx_intvs]...)
  sec_acxmax = vcat([acxi[2] for acxi in intv_info.acx_intvs]...)
  smin, smax = findSectorMinMax(sec_acxmin, sec_acxmax, ffnet.activ)
  qc_sector = QcActivSector(acxdim=acxdim, β=β, smin=smin, smax=smax, base_smin=0.0, base_smax=1.0)

  qc_activs = [qc_bounded; qc_sector]
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



