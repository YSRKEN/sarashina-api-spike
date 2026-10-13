def next_bus(times, now):
    t=sorted(times)
    for x in t:
        if x>=now: return x
    return t[0]
