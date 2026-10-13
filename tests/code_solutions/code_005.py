def stair_ways(n, broken):
    M=10**9+7; b=set(broken); d=[0]*(n+1); d[0]=1
    for i in range(1,n+1):
        if i in b: continue
        d[i]=(d[i-1]+(d[i-2] if i>=2 else 0))%M
    return d[n]
