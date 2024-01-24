from collections import Counter, defaultdict

def prob(s):
        maxL = 0
        chSet = defaultdict(set)
        properWords = {}
        goldenWords = {}
        for i,word in enumerate(s):
            if len(set(word)) != len(word): continue
            for c in word:  chSet[c].add(i)
        
        return maxL


questions = [
    ["un","iq","ue"]
]

for s in questions:
    print(prob(s))