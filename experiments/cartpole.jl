start_time = time()

using LinearAlgebra
using Distributions
using ArgParse
using Printf
using Dates
using DataFrames
using CSV
using Plots

include("../src/NnSdp.jl"); using .NnSdp
const nn = NnSdp

@printf("load done: %.3f\n", time() - start_time)

DUMP_DIR = joinpath(@__DIR__, "..", "dump", "cartpole")

mosek_opts =
  Dict("QUIET" => false,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 3, # seconds
       "MSK_IPAR_INTPNT_SCALING" => 1,
       "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-8,
       "INTPNT_CO_TOL_REL_GAP" => 1e-6,
       "INTPNT_CO_TOL_PFEAS" => 1e-6,
       "INTPNT_CO_TOL_DFEAS" => 1e-6)

DOPTS = DeepSdpOptions(use_dual=true, mosek_opts=mosek_opts, verbose=true)
COPTS = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:single_decomp)
C2OPTS = ChordalSdpOptions(mosek_opts=mosek_opts, verbose=true, decomp_mode=:double_decomp)

x1min = [2.000; 1.000; -0.174; -1.000]
x1max = [2.200; 1.200; -0.104; -0.800]

#=
x1center = [2.1; 1.1; -0.135; -0.9]
x1min = x1center .- 1e-4
x1max = x1center .+ 1e-4
=#


makeCartpole(t) = load(joinpath(@__DIR__, "..", "models", "cartpole$(t).pth"))
ffnet_cartpole = makeCartpole(1)
ts = [1; 2; 3; 4; 5; 6; 7; 8]
# dims = [1; 2]
# βs = [0; 1]

opts2string(opts::DeepSdpOptions) = "deepsdp" * (if opts.use_dual; "__dual" else "" end)
opts2string(opts::ChordalSdpOptions) = "chordal" * (if opts.use_dual; "__dual" else "" end) * "__$(opts.decomp_mode)"

function cartpole(z::Vector, F::Real; m_cart=0.25, m_pole=0.1, l=0.4, g=9.81)
  @assert length(z) == 4
  x, dx, θ, dθ = z[1], z[2], z[3], z[4]
  M = m_pole + m_cart
  ddθ_top = g*sin(θ) + cos(θ) * ((-F - m_pole * l * dθ^2 * sin(θ)) / M)
  ddθ_bot = l * ((4/3) - (m_pole * cos(θ)^2 / M))
  ddθ = ddθ_top / ddθ_bot
  ddx_top = F + m_pole * l * (dθ^2 * sin(θ) - ddθ*cos(θ))
  ddx = ddx_top / M
  dz = [dx; ddx; dθ; ddθ]
  return dz
end

function cartpoleTraj(z1::Vector, T; dt = 0.05)
  zs = [z1]
  zt = z1
  for t in 2:T
    dzt = cartpole(zt, 0.0)
    zt = zt + dt*zt
    push!(zs, zt)
  end
  return zs
end

function randx1(z1min, z1max)
  x1 = rand(Uniform(z1min[1], z1max[1]))
  x2 = rand(Uniform(z1min[2], z1max[2]))
  x3 = rand(Uniform(z1min[3], z1max[3]))
  x4 = rand(Uniform(z1min[4], z1max[4]))
  z1 = Vector{Float64}([x1; x2; x3; x4])
  return z1
end

function randomCartpoleTrajs(z1min, z1max, T; N = 1000)
  trajs = Vector{Any}()
  for n in 1:N
    z1 = randx1(z1min, z1max)
    traj = cartpoleTraj(z1, T)
    push!(trajs, traj)
  end
  return trajs
end

function runFeedFwdNetTraj(ffnet, x1, T)
  xs, xt = [x1], x1
  for t in 2:T
    xt = evalNet(ffnet, xt)
    push!(xs, xt)
  end
  return xs
end

function randomNeuralCartpoleTrajs(z1min, z1max, T; N = 1000)
  trajs = Vector{Any}()
  for n in 1:N
    z1 = randx1(z1min, z1max)
    traj = runFeedFwdNetTraj(ffnet_cartpole, z1, T)
    push!(trajs, traj)
  end
  return trajs
end

# Neural trajectory of cartpole

# Run a single β, dim pair
function go(β, dim, opts; dosave = true)
  saveto = joinpath(DUMP_DIR, "cartpole_beta$(β)_dim$(dim)_$(opts2string(opts)).csv")
  printstyled("running with β: $(β) at dim $(dim) | now is: $(now())\n", color=:green)
  qc_input = QcInputBox(x1min=x1min, x1max=x1max)
  df = DataFrame(t = Int[],
                 pos_val = Real[],
                 neg_val = Real[],
                 pos_setup_secs = Real[],
                 neg_setup_secs = Real[],
                 pos_solve_secs = Real[],
                 neg_solve_secs = Real[],
                 pos_total_secs = Real[],
                 neg_total_secs = Real[],
                 pos_term_status = String[],
                 neg_term_status = String[],
                 pos_eigmax = Real[],
                 neg_eigmax = Real[])
  for t in ts
    printstyled("\tβ: $(β), dim: $(dim), t: $(t) | now is: $(now())\n", color=:green)
    ffnet = makeCartpole(t)
    qc_activs = makeQcActivs(ffnet, x1min=x1min, x1max=x1max, β=β)

    printstyled("\t\tpositive:\n", color=:green)
    query_pos = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = QcReachHplane(normal=Vector(e(dim,4))),
                           obj_func = x -> x[1])
    soln_pos = Methods.runQuery(query_pos, opts)
    λmax_pos = eigmax(Symmetric(Matrix(soln_pos.values[:Z])))

    printstyled("\t\tnegative:\n", color=:green)
    query_neg = ReachQuery(ffnet = ffnet,
                           qc_input = qc_input,
                           qc_activs = qc_activs,
                           qc_reach = QcReachHplane(normal=Vector(-e(dim,4))),
                           obj_func = x -> x[1])
    soln_neg = Methods.runQuery(query_neg, opts)
    λmax_neg = eigmax(Symmetric(Matrix(soln_neg.values[:Z])))
    entry = (t,
             soln_pos.objective_value, 
             soln_neg.objective_value,
             soln_pos.setup_time,
             soln_neg.setup_time,
             soln_pos.solve_time,
             soln_neg.solve_time,
             soln_pos.total_time,
             soln_neg.total_time,
             soln_pos.termination_status,
             soln_neg.termination_status,
             λmax_pos,
             λmax_neg)
    push!(df, entry)
    if dosave
      CSV.write(saveto, df)
      printstyled("updated $(saveto)\n", color=:green)
    end
  end
end

function goAll()
  # β = 0
  go(0, 1, DOPTS)
  go(0, 2, DOPTS)
  go(0, 3, DOPTS)
  go(0, 4, DOPTS)

  # β = 1
  go(1, 1, C2OPTS)
  go(1, 2, C2OPTS)
  go(1, 3, C2OPTS)
  go(1, 4, C2OPTS)

  # β = 2
  go(2, 1, C2OPTS)
  go(2, 2, C2OPTS)
  go(2, 3, C2OPTS)
  go(2, 4, C2OPTS)

  # β = 3
  go(3, 1, C2OPTS)
  go(3, 2, C2OPTS)
  go(3, 3, C2OPTS)
  go(3, 4, C2OPTS)

  # β = 4
  go(4, 1, C2OPTS)
  go(4, 2, C2OPTS)
  go(4, 3, C2OPTS)
  go(4, 4, C2OPTS)
end


qc_input = QcInputBox(x1min=x1min, x1max=x1max)
qc_activs = makeQcActivs(ffnet_cartpole, x1min=x1min, x1max=x1max, β=2)

query_pos = ReachQuery(ffnet = ffnet_cartpole,
                       qc_input = qc_input,
                       qc_activs = qc_activs,
                       qc_reach = QcReachHplane(normal=Vector(e(1,4))),
                       obj_func = x -> x[1])


function quickreach(normal)
  query = ReachQuery(ffnet = ffnet_cartpole,
                     qc_input = qc_input,
                     qc_activs = qc_activs,
                     qc_reach = QcReachHplane(normal=normal),
                     obj_func = x -> x[1])
  return query
end

normal = [0; -1; 0; 0]

query_neg = ReachQuery(ffnet = ffnet_cartpole,
                       qc_input = qc_input,
                       qc_activs = qc_activs,
                       # qc_reach = QcReachHplane(normal=Vector(-e(1,4))),
                       qc_reach = QcReachHplane(normal=normal),
                       obj_func = x -> x[1])
# soln_neg = Methods.runQuery(query_neg, opts)

intvs = makeIntervalsInfo(x1min, x1max, ffnet_cartpole)
ymin, ymax = intvs.x_intvs[end]

query_1_pos = quickreach([1;0;0;0])
query_1_neg = quickreach([-1;0;0;0])

query_2_pos = quickreach([0;1;0;0])
query_2_neg = quickreach([0;-1;0;0])

query_3_pos = quickreach([0;0;1;0])
query_3_neg = quickreach([0;0;-1;0])


