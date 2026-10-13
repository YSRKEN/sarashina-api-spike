def merge_slots(slots):
    out=[]
    for a,b in sorted(slots):
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
