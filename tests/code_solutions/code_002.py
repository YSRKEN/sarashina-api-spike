def count_vowel_runs(s):
    n=0; prev=False
    for c in s:
        v=c in 'aeiou'
        if v and not prev: n+=1
        prev=v
    return n
