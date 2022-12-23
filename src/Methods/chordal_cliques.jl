# A clique
const Clique = VecInt

# Finely decomposed info of each Ck: Ck, [Ck1, Ck2], [Dk1, Dk2], which are:
# Ck: the indices within the big Z
# Ckparts: Ck = Ck1 union Ck2, which we always do for k < p
# Dk1, Dk2: the indices within Ckdim
const CkInfo = Tuple{Clique, Vector{Clique}, Vector{Clique}}

# A 2-layer decomposition relative to β
# We want:
#   Z = sum Eck' * Zk * Eck
#   Zk = sum Eckj' * Ykj * Eckj
# Each Yk has size (nk + n{k+1} + β + nK + 1) and is block-arrow shaped
# Make the cliques
function makeCliques(qcs::Vector{QcInfo}, ffnet::FeedFwdNet)
  # Summation function
  S(k) = (k == 0) ? 0 : sum(ffnet.xdims[1:k])

  # Depending on if we're using sectors, adjust β accordingly
  qc_secs = filter(qc -> qc isa QcActivSector, qcs)
  β = (length(qc_secs) == 0) ? 0 : qc_secs[1].β

  if β == 0
    return makeCliquesβ0(ffnet)
  end

  # Now we can begin the clique construction proper
  p = 1
  for i in 1:ffnet.K
    # The first clique to touch the Kth block column is the final clique
    # This is equivalent to having the clique >= S(K-1)
    if S(i+1) + β >= S(ffnet.K-1); p = i; break end
  end

  # Outer cliques Cks, and inner cliques Djs
  cliques = Vector{CkInfo}()
  for k in 1:p-1
    # First calculate the big Ck clique
    Ck1 = VecInt(S(k-1)+1 : S(k+1)+β)
    Ck2 = VecInt(S(ffnet.K-1)+1 : S(ffnet.K)+1)
    @assert Ck1[end] <= Ck2[1]
    Ck = [Ck1; Ck2]
    Ckdim = length(Ck)

    # Now calculate the Djs of each Ck; if k == 1 there is no two-stage decomp
    if k == 1
      Dk1 = VecInt(1 : Ckdim)
      push!(cliques, (Ck, [Ck1, Ck2], [Dk1]))

    # Otherwise do the two-stage decomposition
    else
      # Dk1 consists of the first nk + n{k+1} + β part and the affine part
      nk, nk1 = ffnet.zdims[k], ffnet.zdims[k+1]
      Dk1 = VecInt([(1 : nk+nk1+β); Ckdim])
      Dk2 = VecInt(nk+nk1+β+1 : Ckdim)
      push!(cliques, (Ck, [Ck1, Ck2], [Dk1, Dk2]))
    end
  end

  # Just copy it over the final inner clique for now
  Cp = VecInt(S(p-1)+1 : S(ffnet.K)+1)
  Dp1 = VecInt(1 : length(Cp))
  push!(cliques, (Cp, [Cp], [Dp1]))
  return cliques
end

# When β == 0, we do some special casing
function makeCliquesβ0(ffnet::FeedFwdNet)
  S(k) = (k == 0) ? 0 : sum(ffnet.xdims[1:k])
  p = ffnet.K-2
  cliques = Vector{CkInfo}()
  for k in 1:p
    Ck1 = VecInt(S(k-1)+1 : S(k+1))
    Ck2 = VecInt(S(ffnet.K-1)+1 : S(ffnet.K)+1)
    Ck = [Ck1; Ck2]
    Ckdim = length(Ck)

    # Ignore the k = 1 clique, similar to before
    if k == 1
      Dk1 = VecInt(1 : Ckdim)
      push!(cliques, (Ck, [Ck1, Ck2], [Dk1]))

      # But every clique afterwards may be decomposed
    else
      nk, nk1 = ffnet.zdims[k], ffnet.zdims[k+1]
      Dk1 = VecInt([(1 : nk+nk1); Ckdim])
      Dk2 = VecInt(nk+nk1+1 : Ckdim)
      push!(cliques, (Ck, [Ck1, Ck2], [Dk1, Dk2]))
    end
  end
  return cliques
end

