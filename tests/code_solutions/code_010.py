def split_bill(total, weights):
    W=sum(weights); base=[total*w//W for w in weights]; rem=[total*w%W for w in weights]
    left=total-sum(base)
    for i in sorted(range(len(weights)), key=lambda i:(-rem[i],i))[:left]: base[i]+=1
    return base
