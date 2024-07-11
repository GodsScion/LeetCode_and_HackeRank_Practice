from collections import defaultdict, Counter

class Solution:

    #######  ARRAYS AND HASHING  #######
    # 217
    def containsDuplicate(self, nums: list[int]) -> bool:
        prev = set()
        for n in nums:
            if n in prev:
                return True
            prev.add(n)
        return False


# Problem 212
class Trie:
    def __init__(self):
        self.children = {}
        self.isWord = False

    def add(self, word: str):
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isWord = True

    def remove(self, word: str):
        cur = self
        nodes = []
        for i, ch in enumerate(word):
            if ch not in cur.children:
                return
            nodes.append(cur)
            cur = cur.children[ch]
        cur.isWord = False
        i = -1
        while nodes:
            parent = nodes.pop()
            if len(cur.children) > 0 or cur.isWord:
                return
            parent.children.pop(word[i])
            cur = parent
            i -= 1


class Solution:
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        found = set()
        
        trie = Trie()
        for word in words: trie.add(word)

        row, col = len(board), len(board[0])
        
        visited = set()
        def search(r, c, parent, visited, word):
            if (r < 0 or c < 0 or r >= row or c >= col or (r,c) in visited or board[r][c] not in parent.children): return

            word += board[r][c]

            parent = parent.children[board[r][c]]
            if parent.isWord:
                found.add(word)
                trie.remove(word)

            visited.add((r,c))

            search(r+1, c, parent, visited, word)
            search(r, c+1, parent, visited, word)
            search(r-1, c, parent, visited, word)
            search(r, c-1, parent, visited, word)
            
            visited.remove((r, c))


        for r in range(row):
            for c in range(col):
                search(r, c, trie, visited, "")

        return list(found)


testCases = [
    ([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"], ["oath","eat"]),
    ([["a","a"]], ["aa"], ["aa"])
]

for board, words, ans in testCases:
    print(Solution().findWords(board, words), ans)



# class Solution:
#     def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        

            



# # Problem 33
# class Solution:
#     def search(self, nums: [int], target: int) -> int:        
#         left, right = 0, len(nums) - 1
        
#         while left <= right:
#             mid = (left + right) // 2
#             if target == nums[mid]: return mid
            
#             if nums[left] <= nums[mid]:
#                 if target > nums[mid] or target < nums[left]:   left = mid + 1
#                 else:   right = mid - 1
#             else:
#                 if target < nums[mid] or target > nums[right]:  right = mid - 1
#                 else:   left = mid + 1
#         return -1
    
# tests = [
#     ([4,5,6,7,0,1,2], 0, 4),
#     ([4,5,6,7,0,1,2], 3, -1),
# ]

# for test in tests:
#     print(Solution().search(test[0],test[1]), test[2])

# Problem 76
# class Solution:
#     def minWindow(self, s: str, t: str) -> str:
#         if len(t) == 0 or len(t) > len(s): return ""
        
#         minSub = s+s
#         T = Counter(t)
#         window = defaultdict(int)

#         left, have, need = 0, 0, len(T)
        
#         for right,ch in enumerate(s):
#             if ch in T:
#                 window[ch]+=1
#                 if window[ch] == T[ch]: have+=1
#                 if have == need: 
#                     while True:
#                         while left<right and s[left] not in T: left+=1
#                         if not window[s[left]] > T[s[left]]:   
#                             if len(minSub) > right-left+1: minSub = s[left:right+1]
#                             have-=1
#                             window[s[left]]-=1
#                             left+=1 
#                             break
#                         window[s[left]]-=1
#                         left+=1    
        
#         return minSub if minSub != s+s else ""

# testcases = [
#     ["ADOBECODEBANC","ABC"],
#     ["ABABUIBWEUIFBBIWE","BWE"],
#     # ["ABAA",0],
#     # ["AAAAA",0]
# ]

# for test in testcases: print(Solution().minWindow(test[0],test[1]))


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
