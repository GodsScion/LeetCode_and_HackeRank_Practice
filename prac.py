from collections import Counter, defaultdict

def prob(nums,k):
        numIndex = {}
        for i, num in enumerate(nums):
            if num in numIndex and i - numIndex[num] <= k:  return True
            numIndex[num] = i
        return False


testCases = [
    ([1,2,3,4,5,6,7,7,8],4)      
]

for num,k in testCases: print(prob(num,k))

# def prob(s):
#         maxL = 0
#         chSet = defaultdict(set)
#         properWords = {}
#         goldenWords = {}
#         for i,word in enumerate(s):
#             if len(set(word)) != len(word): continue
#             for ch in word:  chSet[ch].add(i)
        
#         return maxL
# questions = [
#     ["un","iq","ue"]
# ]

# for s in questions:
#     print(prob(s))
