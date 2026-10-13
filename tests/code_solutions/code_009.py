def bracket_depth(s):
    pair={')':'(',']':'[','}':'{'}; st=[]; d=0
    for c in s:
        if c in '([{': st.append(c); d=max(d,len(st))
        elif c in pair:
            if not st or st.pop()!=pair[c]: return -1
    return -1 if st else d
