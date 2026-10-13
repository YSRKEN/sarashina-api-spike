def rle_decode(s):
    out=[]; num=''
    for c in s:
        if c.isdigit(): num+=c
        else: out.append(c*(int(num) if num else 1)); num=''
    return ''.join(out)
