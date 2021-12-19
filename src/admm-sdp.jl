module AdmmSdp

using ..Header
using ..Common
using ..Intervals
using Parameters
using LinearAlgebra
using JuMP
using Mosek
using MosekTools

@with_kw mutable struct AdmmParams
  γ :: Vector{Float64}
  vs :: Vector{Vector{Float64}}
  ωs :: Vector{Vector{Float64}}
  λs :: Vector{Vector{Float64}}
  μs :: Vector{Vector{Float64}}
end

@with_kw struct AdmmCache
  Js :: Vector{Matrix{Float64}}
  zaffs :: Vector{Vector{Float64}}
  Jtzaffs :: Vector{Vector{Float64}}
  I_JtJ_invs :: Vector{Matrix{Float64}}
end

@with_kw struct AdmmSdpOptions
  max_iters :: Int = 400
  begin_check_at_iter :: Int = 5
  check_every_k_iters :: 2
  nsd_tol :: Float64 = 1e-4
  ρ :: Float64 = 1.0
  verbose :: Bool = false
end

end # End module

