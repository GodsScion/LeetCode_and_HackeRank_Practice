def prob(s,k):
        maxL = 0
        l = 0
        while l < len(s):
            tempK = k
            r = l
            while r < len(s) and s[r] == s[l]:  r+=1
            newC = r
            while tempK > 0 and r < len(s) and s[r] != s[l]:
                r += 1
                tempK -= 1
            while r < len(s) and s[r] == s[l]:
                while r < len(s) and s[r] == s[l]:  r+=1
                while r < len(s) and s[r] != s[l] and tempK > 0:
                    r += 1
                    tempK -= 1
            maxL = max(maxL, min(r-l+tempK, len(s)) )
            l = newC
        return maxL


questions = [
    ("ABAB",2),
    ("AABABBA",1)
]

for s,a,b,k in questions:
    print(prob(s,a,b,k))