# A 2-layer decomposition relative to β
# We want:
#   Z = sum Eck' * Zk * Eck
#   Zk = sum Eckj' * Ykj * Eckj
# Each Yk has size (nk + n{k+1} + β + nK + 1) and is block-arrow shaped
function makeStage2Cliques(β::Int, ffnet::FeedFwdNet)
  S(k) = (k == 0) ? 0 : sum(ffnet.zdims[1:k])
  p = 1
  for i in 1:ffnet.K
    # The first clique to touch the Kth block column is the final clique
    # This is equivalent to having the clique >= S(K-1)
    if S(i+1) + β >= S(ffnet.K-1); p = i; break end
  end

  # Outer cliques Cks, and inner cliques Djs
  cliques = Vector{Tuple{Clique, Vector{Clique}}}()
  for k in 1:p-1
    # First calculate the big Ck clique
    Ck_init1 = S(k-1) + 1
    Ck_initdim = ffnet.zdims[k] + ffnet.zdims[k+1] + β
    Ck_init = Ck_init1 : (Ck_init1 + Ck_initdim - 1)
    Ck_tail1 = S(ffnet.K-1) + 1
    Ck_taildim = ffnet.zdims[ffnet.K] + 1
    Ck_tail = Ck_tail1 : (Ck_tail1 + Ck_taildim - 1)
    @assert Ck_init[end] <= Ck_tail[1]
    Ck = VecInt([Ck_init; Ck_tail])
    Ckdim = length(Ck)

    # Now calculate the Djs of each Ck
    # If k = 1, 
    if k == 1
      Dk1 = VecInt(1:Ckdim)
      push!(cliques, (Ck, [Dk1]))

    # Otherwise do the two-stage decomposition
    else
      # Dk1 consists of the first nk + n{k+1} + β part and the affine part
      Dk1_init = 1 : Ck_initdim
      Dk1_last = Ckdim
      Dk1 = VecInt([Dk1_init; Dk1_last])

      # Dk2 begins after nk + n{k+1} + β and goes until the end
      Dk2 = VecInt(Ck_initdim+1 : Ckdim)
      push!(cliques, (Ck, [Dk1, Dk2]))
    end
  end

  # Special case the last clique
  Cp_init1 = S(p-1) + 1
  Cp_dim = sum(ffnet.zdims) - S(p-1)
  Cp = VecInt(Cp_init1 : (Cp_init1 + Cp_dim - 1))

  # Just copy it over the final inner clique for now
  Dp1 = VecInt(1:length(Cp))
  push!(cliques, (Cp, [Dp1]))
  return cliques
end

# Check that this is a valid qcs
# Must have at least one QcInput
# Must have at least one QcActiv
# Must have exactly one QcOutput
function isValidQcInfos(qcs::Vector{QcInfo})
  qc_ins = filter(qc -> qc isa QcInput, qcs)
  qc_acs = filter(qc -> qc isa QcActiv, qcs)
  qc_outs = filter(qc -> qc isa QcOutput, qcs)
  return (length(qc_ins) >= 1 && length(qc_acs) >= 1 && length(qc_outs) == 1)
end

# Given a bunch of qcs, return a Vector{VecInt} of their cliques
function makeCliques(qcs::Vector{QcInfo}, ffnet::FeedFwdNet)
  # Check validity, and that QcActivSector is among them
  @assert isValidQcInfos(qcs)

  # Depending on if we're using sectors, adjust β accordingly
  qc_secs = filter(qc -> qc isa QcActivSector, qcs)
  β = (length(qc_secs) == 0) ? 0 : qc_secs[1].β

  return makeStage2Cliques(β, ffnet)
end

