def tally_votes(votes):
    cnt={}
    for v in votes: cnt[v]=cnt.get(v,0)+1
    return sorted(cnt, key=lambda k:-cnt[k])
