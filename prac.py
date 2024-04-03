from collections import defaultdict, Counter

# Problem 76
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) == 0 or len(t) > len(s): return ""
        
        minSub = s+s
        T = Counter(t)
        window = defaultdict(int)

        left = 0
        while left<len(s) and s[left] not in T: left+=1
        
        for right in range(left,len(s)):
            if s[right] in T:
                window[s[right]]+=1
                if T == window: 
                    while True:
                        while left<right and s[left] not in T: left+=1
                        if window[s[left]] > T[s[left]]:
                            window[s[left]]-=1
                            left+=1
                        else:   
                            if len(minSub) > right-left+1: minSub = s[left:right+1]
                            break
        
        return minSub if minSub != s+s else ""

testcases = [
    ["ADOBECODEBANC","ABC"],
    ["ABABUIBWEUIFBBIWE","BWE"],
    # ["ABAA",0],
    # ["AAAAA",0]
]

for test in testcases: print(Solution().minWindow(test[0],test[1]))


# Problem 424
# def characterReplacement( s: str, k: int) -> int:
#     maxLen = 0
#     maxFreq = 0
#     window = defaultdict(int)
#     left = 0
#     for right,ch in enumerate(s):
#         window[ch] += 1
#         maxFreq = max(maxFreq, window[ch])
#         if right - left + 1 > maxFreq + k:
#             window[s[left]] -= 1
#             left += 1
#         maxLen = max(maxLen, right-left)
#     return maxLen

# testCases = [
#     ('ABAB',2),
#     ('ABAAABBAAAA',2)
# ]

# for s,k in testCases: print(characterReplacement(s, k))




# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# # class Solution:
# def addTwoNumbers( l1: ListNode, l2: ListNode) -> ListNode:
#         def addNodes(l1,l2,c):john
#             v = l2.val + l1.val
#             if v > 9:
#                 l2.val = v - 10 + c
#                 c = 1
#             else:
#                 l2.val = v + c
#                 c = 0
#             return c

#         c = 0
#         output = l2
#         while l2.next and l1.next:
#             c = addNodes(l1,l2,c)
#             l1 = l1.next
#             l2 = l2.next
#         c = addNodes(l1,l2,c)
#         if l1.next: l2.next = l1.next
#         if l2.next and c:
#             l2 = l2.next
#             while l2.next and c:
#                 if l2.val == 9:
#                     l2.val = 0
#                 else:
#                     l2.val += 1
#                     c = 0
#                 l2 = l2.next
#             if c:   
#                 if l2.val == 9:
#                     l2.val = 0
#                 else:
#                     l2.val += 1
#                     c = 0
#         if c:   l2.next = ListNode(val = 1)
#         return output

# def buildNode(l):
#     temp = ListNode()
#     L = temp
#     for v in l:
#         temp.val = v
#         temp.next = ListNode()
#         temp = temp.next
#     temp = None
#     return L
        
# testCases = [
#     ([3,7],[9,2])
# ]

# for l1,l2 in testCases: print(addTwoNumbers(buildNode(l1),buildNode(l2)))

# from collections import Counter, defaultdict

# def prob(nums,k):
#         numIndex = {}
#         for i, num in enumerate(nums):
#             if num in numIndex and i - numIndex[num] <= k:  return True
#             numIndex[num] = i
#         return False


# testCases = [
#     ([1,2,3,4,5,6,7,7,8],4)      
# ]

# for num,k in testCases: print(prob(num,k))

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
