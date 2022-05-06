
# Plot a bunch of 2D points and return the modified plot object
function plotScatterPlots!(plt, points::Vector{<:VecReal}; kwargs...)
  @assert all(p -> length(p) == 2, points)
  xs, ys = [p[1] for p in points], [p[2] for p in points]
  scatter!(plt, xs, ys; kwargs...)
	return plt
end

# Plot a sequence of 2D points
function plotSeqPoints!(plt, points::Vector{<:VecReal}; kwargs...)
  @assert all(p -> length(p) == 2, points)
  xs, ys = [p[1] for p in points], [p[2] for p in points]
  plot!(plt, xs, ys; kwargs...)
  return plt
end

const Hplane = Tuple{<:VecReal, <:Real}
const Polygon = Vector{<:Hplane}

# Plot a 2D plot polygon defined as a sequence of hyperplanes
function plotPolygon!(plt, poly::Polygon; kwargs...)
  hplanes = poly
  @assert all(hp -> length(hp[1]) == 2, hplanes)
  augs = [hplanes; hplanes[1]; hplanes[2]]
  verts = Vector{VecReal}()
  for i in 1:(length(augs)-1)
    # Normal (n) and offset (h)
    n1, h1 = augs[i]
    n2, h2 = augs[i+1]
    x = [n1'; n2'] \ [h1; h2]
    push!(verts, x)
  end
  plt = plotSeqPoints!(plt, verts; kwargs...)
  return plt
end

# Plot a bunch of points and the respective polytopes
function plotBoundingPolygons!(plt, points::Vector{<:VecReal}, polys::Vector{<:Polygon}; kwargs...)
  @assert all(p -> length(p) == 2, points) # All points are 2D
  @assert all(hp -> length(hp[1]) == 2, polys) # All hplane normals are 2D
  @assert length(polys) >= 1 # At least one poly

  colors = theme_palette(:auto)
  plt = plotScatterPlots!(plt, points, color=colors[1], alpha=0.3)

  for (i, poly) in enumerate(polys)
    plt = plotPolygon!(plt, poly, color=colors[i+1])
  end

  # Figure out the bounding regions
  pxs = [p[1] for p in points]
  polyxs = vcat([hp[1][1] for hp in polys]...)
  allxs = [pxs; polyxs]

  pys = [p[2] for p in points]
  polyys = vcat([hp[1][2] for hp in polys]...)
  allys = [pys; polyys]

  xmin, xmax = minimum(allxs), maximum(allxs)
  ymin, ymax = minimum(allys), maximum(allys)
  xgap, ygap = (xmax-xmin), (ymax-ymin)
  xlim = (xmin - 0.1 * xgap, xmax + 0.1 * xgap)
  ylim = (ymin - 0.1 * ygap, ymax + 0.1 * ygap)
  plt = plot!(plt, xlim=xlim, ylim=ylim)
  return plt
end

# We accept general square matrices here to make life easier
const Ellipse = Tuple{<:MatReal, <:VecReal}

# Plot the boundary of a 2D ellpse: {y = Px + yc : ||x||^2 = 1}
function plotEllipse!(plt, ellipse::Ellipse; kwargs...)
  P, yc = ellipse
  @assert size(P) == (2, 2)
  @assert length(yc) == 2
  c(t) = P * [cos(t); sin(t)] + yc
  c1(t) = c(t)[1]
  c2(t) = c(t)[2]
  ts = range(0, stop=2*π, length=10000)
  plt = plot!(plt, c1.(ts), c2.(ts))
  # plt = plot!(plt, c1, c2, ts)
  return plt
end

# Plot a bunch of points and the respective ellipses
function plotBoundingEllipses!(plt, points::Vector{<:VecReal}, ellipses::Vector{<:Ellipse}; kwargs...)
  @assert all(p -> length(p) == 2, points) # All points are 2D
  @assert all(ellip -> size(ellip[1]) == (2,2), ellipses)
  @assert all(ellip -> length(ellip[2]) == 2, ellipses)
  @assert length(ellipses) >= 1 # At least one ellipse

  colors = theme_palette(:auto)
  plt = plotScatterPlots!(plt, points, color=colors[1], alpha=0.3)

  for (i, ellip) in enumerate(ellipses)
    plt = plotEllipse!(plt, ellip, color=colors[i+1])
  end

  # Figure out the bounding regions
  λmaxs = [sqrt(eigmax(P' * P)) for (P, _) in ellipses]
  pxs = [p[1] for p in points]
  ellipxmins = [yc[1] - (λmaxs[k]) for (k, (_, yc)) in enumerate(ellipses)]
  ellipxmaxs = [yc[1] + (λmaxs[k]) for (k, (_, yc)) in enumerate(ellipses)]
  allxs = [pxs; ellipxmins; ellipxmaxs]

  pys = [p[2] for p in points]
  ellipymins = [yc[2] - (λmaxs[k]) for (k, (_, yc)) in enumerate(ellipses)]
  ellipymaxs = [yc[2] + (λmaxs[k]) for (k, (_, yc)) in enumerate(ellipses)]
  allys = [pys; ellipymins; ellipymaxs]

  xmin, xmax = minimum(allxs), maximum(allxs)
  ymin, ymax = minimum(allys), maximum(allys)
  xgap, ygap = (xmax-xmin), (ymax-ymin)
  xlim = (xmin - 0.1 * xgap, xmax + 0.1 * xgap)
  ylim = (ymin - 0.1 * ygap, ymax + 0.1 * ygap)

  plt = plot!(plt, xlim=xlim, ylim=ylim)
  return plt
end



