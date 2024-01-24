from collections import Counter, defaultdict

def prob(nums,k):
        i = 0
        window = set()
        for i in range(min(k,len(nums))):
            if nums[i] in window: return True
            


testCases = [
    ([1,2,3,4,5,6,7,7,8],5)      
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
