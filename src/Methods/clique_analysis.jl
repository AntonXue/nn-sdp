function _sectorCliques(qcinfo::QcSectorInfo, ffnet::FeedFwdNet)
  S(k) = (k == 0) ? 0 : sum(ffnet.zdims[1:k])
  p = 1
  for i in 1:ffnet.K
    if S(i+1) + qcinfo.β >= S(ffnet.K-1); p = i; break end
  end

  cliques = Vector{VecInt}()
  # Handle the cliques for k < p
  for k in 1:p-1
    Ck_init1 = S(k-1) + 1
    Ck_initdim = ffnet.zdims[k] + ffnet.zdims[k+1] + qcinfo.β
    Ck_init = Ck_init1 : (Ck_init1 + Ck_initdim - 1)
    Ck_tail1 = S(ffnet.K-1) + 1
    Ck_taildim = ffnet.zdims[ffnet.K] + 1
    Ck_tail = Ck_tail1 : (Ck_tail1 + Ck_taildim - 1)
    @assert Ck_init[end] <= Ck_tail[1]
    Ck = VecInt([Ck_init; Ck_tail])
    push!(cliques, Ck)
  end

  # Now handle the last clique
  Cp_init1 = S(p-1) + 1
  Cp_dim = sum(ffnet.zdims) - S(p-1)
  Cp = VecInt(Cp_init1 : (Cp_init1 + Cp_dim - 1))
  push!(cliques, Cp)

  # And return
  return cliques
end

# A 2-layer decomosition
function sectorCliques(qcinfo::QcSectorInfo, ffnet::FeedFwdNet)
  S(k) = (k == 0) ? 0 : sum(ffnet.zdims[1:k])
  p = 1
  for i in 1:ffnet.K
    if S(i+1) + qcinfo.β >= S(ffnet.K-1); p = i; break end
  end

  Clique = VecInt
  # Outer cliques Ck, and inner cliques Dkl
  cliques = Vector{Tuple{Clique, Vector{Clique}}}()
  for k in 1:p-1
    # First calculate the big Ck clique
    Ck_init1 = S(k-1) + 1
    Ck_initdim = ffnet.zdims[k] + ffnet.zdims[k+1] + qcinfo.β
    Ck_init = Ck_init1 : (Ck_init1 + Ck_initdim - 1)
    Ck_tail1 = S(ffnet.K-1) + 1
    Ck_taildim = ffnet.zdims[ffnet.K] + 1
    Ck_tail = Ck_tail1 : (Ck_tail1 + Ck_taildim - 1)
    @assert Ck_init[end] <= Ck_tail[1]
    Ck = VecInt([Ck_init; Ck_tail])

    # Now calculate the sub-cliques
    Dcliques = Vector{Clique}()
    # In the first case, there is nothing special
    if k == 1
      Dk1 = VecInt(1:length(Ck))
      push!(cliques, (Ck, [Dk1]))

    # Otherwise, only use the first n[k] + n[k+1] + β, and the affine part
    else
      Dk1_init = 1 : Ck_initdim
      Dk1_last = length(Ck)
      Dk1 = VecInt([Dk1_init; Dk1_last])
      push!(cliques, (Ck, [Dk1]))
    end
  end

  # Special case the last clique
  Cp_init1 = S(p-1) + 1
  Cp_dim = sum(ffnet.zdims) - S(p-1)
  Cp = VecInt(Cp_init1 : (Cp_init1 + Cp_dim - 1))

  # Just copy it over for now
  Dp1 = VecInt(1:length(Cp))
  push!(cliques, (Cp, [Dp1]))

  return cliques
end

# Given a bunch of qcinfos, return a Vector{VecInt} of their cliques
function findCliques(qcinfos::Vector{QcInfo}, ffnet::FeedFwdNet)
  # Assume that QcSectorInfo is among them
  @assert any(qi -> qi isa QcSectorInfo, qcinfos)

  # For now, use hte cliques returned by the sector with the banded T
  for qcinfo in qcinfos
    if qcinfo isa QcSectorInfo
      sector_cliques = sectorCliques(qcinfo, ffnet)
      return sector_cliques
    end
  end
  error("did not have any QcSectorInfos")
end

