
# Plot a bunch of 2D points and return the modified plot object
function plotScatterPlots!(plt, points::Vector{<:VecReal}; kwargs...)
  @assert all(p -> length(p) == 2, points)
  xs, ys = [p[1] for p in points], [p[2] for p in points]
  scatter!(plt, xs, ys; kwargs...)
  return plt
end

# Plot a sequence of 2D points
function plotSeq2DPoints!(plt, points::Vector{<:VecReal}; kwargs...)
  @assert all(p -> length(p) == 2, points)
  xs, ys = [p[1] for p in points], [p[2] for p in points]
  plot!(plt, xs, ys; kwargs...)
  return plt
end

