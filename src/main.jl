
include("header.jl")
include("common.jl")
include("deep-sdp.jl")

using .Header
using .Common
using .DeepSDP


A = randn(4, 6)
B = randn(3, 5)
C = randn(2, 4)
D = randn(1, 3)

M = [A, B, C, D]

xd = [5, 4, 3, 2, 1]

ffnet = FeedForwardNetwork(nettype=ReluNetwork(), xdims=xd, M=M)

Q1 = randn(xd[2] + xd[2] + 1, xd[2] + xd[2] + 1)
Q2 = randn(xd[3] + xd[3] + 1, xd[3] + xd[3] + 1)
Q3 = randn(xd[4] + xd[4] + 1, xd[4] + xd[4] + 1)
Q4 = randn(xd[5] + xd[5] + 1, xd[5] + xd[5] + 1)

xbot = -2 * ones(xd[1])
xtop = 2 * ones(xd[1])
box = BoxConstraint(xbot=xbot, xtop=xtop)

S = randn(xd[end-1] + xd[1] + 1, xd[end-1] + xd[1] + 1)
safety = SafetyConstraint(S)

L = randn(4, 4)
nu = randn(4)
eta = randn(4)


soln = DeepSDP.run(ffnet, box, safety)
;


