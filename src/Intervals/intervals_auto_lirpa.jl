using PyCall

# Set up auto_lirpa_bridge stuff
AUTO_LIRPA_PARENT_DIR = joinpath(@__DIR__, "..", "..", "exts")
if !(AUTO_LIRPA_PARENT_DIR in PyVector(pyimport("sys")."path"))
  pushfirst!(PyVector(pyimport("sys")."path"), AUTO_LIRPA_PARENT_DIR)
end

auto_lirpa_bridge = pyimport("auto_lirpa_bridge")

# Interval propagation with auto_LiRPA
function intervalsAutoLirpa(x1min::VecF64, x1max::VecF64, ffnet::FeedFwdNet)
  @assert length(x1min) == length(x1max)

  # Write the ffnet to an onnx file
  onnx_file = tempname()
  writeOnnx(ffnet, onnx_file)
  x_intvs, _ = auto_lirpa_bridge.find_bounds(onnx_file, x1min, x1max)

  # Do the acx stuff
  acx_intvs = Vector{PairVecF64}()
  for k in 1:ffnet.K-1
    Wk, bk = ffnet.Ms[k][:, 1:end-1], ffnet.Ms[k][:, end]
    xkmin, xkmax = x_intvs[k]
    ykmin = (max.(Wk, 0) * xkmin) + (min.(Wk, 0) * xkmax) + bk
    ykmax = (max.(Wk, 0) * xkmax) + (min.(Wk, 0) * xkmin) + bk
    push!(acx_intvs, (ykmin, ykmax))
  end

  return IntervalsInfo(ffnet=ffnet, x_intvs=x_intvs, acx_intvs=acx_intvs)
end

