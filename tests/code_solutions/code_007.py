def grid_islands(grid):
    H=len(grid); W=len(grid[0]) if H else 0; seen=set(); out=[]
    for i in range(H):
        for j in range(W):
            if grid[i][j]=='#' and (i,j) not in seen:
                st=[(i,j)]; seen.add((i,j)); n=0
                while st:
                    y,x=st.pop(); n+=1
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        a,b=y+dy,x+dx
                        if 0<=a<H and 0<=b<W and grid[a][b]=='#' and (a,b) not in seen:
                            seen.add((a,b)); st.append((a,b))
                out.append(n)
    return sorted(out, reverse=True)
