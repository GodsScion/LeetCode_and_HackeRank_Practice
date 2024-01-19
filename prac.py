def prob(s,a,b,k):
        i = 0
        j = -1
        start = 0
        sol = []
        while True:
            i = s.find(a,start)
            if i==-1:   break
            start = i+1
            j = s.find(b,max(i-k,0),i+k+len(b))
            if j > -1:
                sol.append(i)
            else:
                j = s.find(b,i+k)
                if j > -1:  start = max(j-k,start)
                else:  break
        return sol 


questions = [
    ("isawsquirrelnearmysquirrelhouseohmy","my","squirrel",15),
    ("dexgscgecd","gscge","d",6),
    ("tdbnme","t","dbnme",4),
    ("vatevavakz","va","lbda",1)
]

for s,a,b,k in questions:
    print(prob(s,a,b,k))