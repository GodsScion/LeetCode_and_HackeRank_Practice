from calendar import c
from types import List, Optional
from collections import defaultdict, Counter
import re
import bisect
from functools import cache

#######  ARRAYS AND HASHING  #######
# 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        prev = set()
        for n in nums:
            if n in prev:
                return True
            prev.add(n)
        return False
# 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
    

# 242. Valid Anagram (https://leetcode.com/problems/valid-anagram/description/) - Easy
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
    

# 1. Two Sum (https://leetcode.com/problems/two-sum/description/) - Easy
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        visited = {}
        for i,num in enumerate(nums):
            if target - num in visited: return [i,visited[target-num]]
            visited[num] = i


# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashMap = dict()
        for word in strs:
            key = "".join(sorted(word))
            if key not in hashMap:
                hashMap[key] = [word]
            else:
                hashMap[key].append(word)
        return hashMap.values()
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashMap = defaultdict(list)
        for word in strs:
            key = [0] * 26
            for ch in word:
                key[ord(ch) - ord('a')] += 1
            key = str(key)
            hashMap[key].append(word)
        return hashMap.values()
# 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        values = {'a':2,'b':3,'c':5,'d':7,'e':11,'f':13,'g':17,'h':19,'i':23,'j':29,'k':31,'l':37,'m':41,'n':43,'o':47,'p':53,'q':59,'r':61,'s':67,'t':71,'u':73,'v':79,'w':83,'x':89,'y':97,'z':101}
        hashSet = defaultdict(list)
        for word in strs:
            h = 1
            for ch in word: h *= values[ch]
            hashSet[h].append(word)
        return hashSet.values()


# 347. Top K Frequent Elements (https://leetcode.com/problems/top-k-frequent-elements/description/) - Medium
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq = Counter(nums)
        buckets = [[] for _ in range(len(nums))]
        for num, count in freq.items():
            buckets[count-1].append(num)
        output = []
        while len(output) != k:
            output += buckets.pop()[0: k-len(output)]
        return output


# 238. Product of Array Except Self (https://leetcode.com/problems/product-of-array-except-self/description/) - Medium
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        p = 1
        result = []
        for num in nums:
            result.append(p)
            p *= num
        p = nums[-1]
        for i in range(len(nums)-2,-1,-1):
            result[i] *= p
            p *= nums[i]
        return result


# 36. Valid Sudoku (https://leetcode.com/problems/valid-sudoku/description/) - Medium
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        boxes = [[set() for _ in range(3)] for _ in range(3)]
        col = [set() for _ in range(9)]
        for i in range(9):
            row = set()
            for j in range(9):
                num = board[i][j]
                if num == ".": continue
                if num in row or num in col[j] or num in boxes[i//3][j//3]: return False
                row.add(num)
                col[j].add(num)
                boxes[i//3][j//3].add(num)
        return True


# 128. Longest Consecutive Sequence (https://leetcode.com/problems/longest-consecutive-sequence/description/) - Medium
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        maxSeqLen = 0
        while maxSeqLen < len(numSet):
            num = numSet.pop()
            longest = num+1
            while longest in numSet:
                numSet.remove(longest)
                longest += 1
            num = num-1
            while num in numSet:
                numSet.remove(num)
                num -= 1
            maxSeqLen = max(maxSeqLen, longest-num-1)
        return maxSeqLen




#######  TWO POINTERS  #######
# 125. Valid Palindrome (https://leetcode.com/problems/valid-palindrome/description/) - Easy
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        left = 0
        right = len(s)-1
        while left < len(s)-1 and not s[left].isalnum(): left += 1
        while right > 0 and not s[right].isalnum(): right -= 1
        while left < right:
            if s[left] != s[right]: return False
            left += 1
            right -= 1
            while left < len(s)-1 and not s[left].isalnum(): left += 1
            while right > 0 and not s[right].isalnum(): right -= 1
        return True
# 125. Valid Palindrome (https://leetcode.com/problems/valid-palindrome/description/) - Easy
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r'[^a-z0-9]+', '', s.lower())
        return s == s[::-1]
# 125. Valid Palindrome (https://leetcode.com/problems/valid-palindrome/description/) - Easy
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = [ch for ch in s.lower() if ch.isalnum()]
        return s == list(reversed(s))
# 125. Valid Palindrome (https://leetcode.com/problems/valid-palindrome/description/) - Easy
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join([ch for ch in s.lower() if ch.isalnum()])
        return s[:len(s)//2] == s[-1:-(len(s)//2)-1:-1]


# 167. Two Sum II - Input Array Is Sorted (https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) - Medium
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers)-1
        while left < right:
            current = numbers[right] + numbers[left]
            if current == target:
                return [left+1, right+1]
            if current > target:
                right -= 1
            else:
                left += 1
        return []
# 167. Two Sum II - Input Array Is Sorted (https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) - Medium
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        '''
        Classic example of over engineering 😑. Don't follow this!
        '''
        left = 0
        right = bisect.bisect_right(numbers, target-numbers[left])-1 
        while right > left:
            need = target - numbers[right]
            left = bisect.bisect_left(numbers, need, left, right)
            if numbers[left] == need:
                return [left+1, right+1]
            right = bisect.bisect_right(numbers, target-numbers[left], left, right)-1
        return []


# 15. 3Sum (https://leetcode.com/problems/3sum/description/) - Medium
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        '''
        Classic example of over engineering 😑.
            
        - Time complexity: `nlog(n) + nk - k^2 : O( n^2 )`.
        Where, `n` and `k` are total number of integers and negative integers respectively in given array, and `0 <= k <= n`.

        - Space complexity: `O( log(n) )`.
        Assuming, output array is not considered.
        '''
        nums.sort()
        mid = bisect.bisect_left(nums, 0)
        end = bisect.bisect_right(nums, -1*min(nums[0]+nums[1],nums[0]))-1
        output = []
        for end in range(end, mid-1, -1):
            if end < len(nums)-1 and nums[end+1] == nums[end]:
                continue
            start = bisect.bisect_left(nums, -nums[end] -nums[end-1])
            while start < mid:
                target = -nums[end] - nums[start]
                if target < nums[start]:
                    break
                if nums[bisect.bisect_right(nums, target, start+2, end)-1] == target:
                    output.append( [nums[start], target, nums[end]] )
                while start < mid and nums[start] == nums[start+1]:
                    start+=1
                start += 1
        if mid < len(nums)-2 and nums[mid+2] == 0:
            output.append([0,0,0])
        return output


# 42. Trapping Rain Water (https://leetcode.com/problems/trapping-rain-water/description/) - Hard
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is the length of `height`
    Two pointers approach is more optimal solution has space complexity of O(1)
    Refer: https://neetcode.io/problems/trapping-rain-water/solution
    '''
    def trap(self, height: List[int]) -> int:
        water = 0
        rightMaxH = [0]*len(height)
        
        curMax = 0
        for i in range(len(height)-1,-1,-1):
            curMax = max(height[i], curMax)
            rightMaxH[i] = curMax
        
        curMax = 0
        for i,h in enumerate(height):
            water += max(0, min(curMax, rightMaxH[i]) - h)
            curMax = max(h, curMax)

        return water




#######  SLIDING WINDOW  #######

# 76. Minimum Window Substring (https://leetcode.com/problems/minimum-window-substring/description/) - Hard
class Solution:
    '''
    Time Complexity: O(|s| + |t|) where |s| and |t| are the lengths of strings s and t respectively.
    Space Complexity: O(1) assuming, output array is not considered.
    '''
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""
        need = Counter(t)
        have = Counter(s)
        for c in need:
            if need[c] > have.get(c, 0):
                return ""

        have = dict()
        needCount = len(t)
        minL, minR = 0, len(s)
        l = 0

        for r, c in enumerate(s):
            amount = have.get(c, 0)
            have[c] = amount + 1
            if c in need and amount < need[c]:
                needCount -= 1
            if needCount == 0:
                while l < r:
                    if s[l] in need and have[s[l]] <= need[s[l]]:
                        break
                    have[s[l]] = have[s[l]] - 1
                    l += 1

                if r+1 - l < minR - minL:
                    minR = r+1
                    minL = l
        return s[minL:minR]

# 76. Minimum Window Substring (https://leetcode.com/problems/minimum-window-substring/description/) - Hard - Duplicate
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""
        
        oStart = 0
        oEnd = len(s) + len(t)
        freq = Counter(t)
        need = len(freq)
        available = defaultdict(int)
        important = []
        i = 0
        
        for end,c in enumerate(s):
            if c in freq:
                important.append(end)
                available[c] += 1
                if available[c] == freq[c]:
                    need -= 1
                if need == 0:
                    while important[i] < end and available[s[important[i]]] > freq[s[important[i]]]:
                        available[s[important[i]]] -= 1
                        i+=1
                    if oEnd-oStart > end-important[i]:
                        oEnd = end
                        oStart = important[i]
                    available[s[important[i]]] -= 1
                    i+=1
                    need += 1
        
        return s[oStart: oEnd+1] if oEnd != len(s) + len(t) else ""
    

# 239. Sliding Window Maximum (https://leetcode.com/problems/sliding-window-maximum/description/) - Hard
from collections import deque
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(k)
    where, n is the size of nums and, k is the size of sliding window.
    This is Monotonic Deque approach, the most optimal solution for this problem.
    NOTE: Common pitfall is to use `>=` instead of `>` for condition `nums[i] > deck[-1]`,
    To make sure this condition `deck[0] == nums[i-k]` doesn't cause trouble for duplicate instances, 
    we should save duplicate instances too, so use `>` not `>=`.
    Refer: https://neetcode.io/problems/sliding-window-maximum/solution
    
    
    !!!  ####  NEEDS EXTRA ATTENTION  ####  !!!
    '''
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        deck = deque()
        for i in range(k):
            while deck and nums[i] > deck[-1]:
                deck.pop()
            deck.append(nums[i])
        output.append(deck[0])
        for i in range(k, len(nums)):
            if deck[0] == nums[i-k]:
                deck.popleft()
            while deck and nums[i] > deck[-1]:
                deck.pop()
            deck.append(nums[i])
            output.append(deck[0])
        return output


# 150. Evaluate Reverse Polish Notation (https://leetcode.com/problems/evaluate-reverse-polish-notation/description/) - Medium
from collections import deque
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    where, n is the length of tokens.
    NOTE: Using `//` or `math.floor()` for division is a Pitfall,
    for negative values like `-0.4` they will return `-1`. So use `int()` 
    For recursive solution, refer: https://neetcode.io/problems/evaluate-reverse-polish-notation/solution
    '''
    def evalRPN(self, tokens: List[str]) -> int:
        stack = deque()
        for token in tokens:
            if token == '+':
                stack.append(stack.pop() + stack.pop())
            elif token == '-':
                prevVal = stack.pop()
                stack.append(stack.pop() - prevVal)
            elif token == '*':
                stack.append(stack.pop() * stack.pop())
            elif token == '/':
                prevVal = stack.pop()
                stack.append(int(stack.pop() / prevVal)) # Use `int()`, not `//` or `math.floor()`, cause we might have negative values
            else:
                stack.append(int(token))
        return stack.pop()



# 567. Permutation in String (https://leetcode.com/problems/permutation-in-string/description/) - Medium
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = Counter(s1)
        have = Counter(s2[:len(s1)-1])
        i = 0
        for c in s2[len(s1)-1:]:
            have[c] = have.get(c, 0) + 1
            if need == have: return True
            have[s2[i]] -= 1
            i += 1
        return need == have



#######  STACK  #######
# 155. Min Stack (https://leetcode.com/problems/min-stack/description/) - Medium
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

class MinStack:

    def __init__(self):
        self.minStack = []
        self.stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.minStack:
            self.minStack.append(min(self.minStack[-1], val))
        else:
            self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop(-1)
        self.minStack.pop(-1)

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]




####### LINKED LIST #######
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 141. Linked List Cycle (https://leetcode.com/problems/linked-list-cycle/description/) - Easy
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        runner = chaser = head
        while runner:
            runner = runner.next
            if not runner:
                return False 
            runner = runner.next
            chaser = chaser.next
            if runner == chaser:
                return True
        return False


# 146. LRU Cache (https://leetcode.com/problems/lru-cache/) - Medium
from collections import OrderedDict
class LRUCache:
    '''
    Time Complexity: O(1) for both get and put operations.
    Space Complexity: O(capacity), where capacity is the maximum number of items that can be stored in the cache.

    We use an OrderedDict in this approach, but it can also be implemented using a combination of a doubly linked list and a hash map.
    '''
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.orderedDict: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.orderedDict:
            return -1
        self.orderedDict.move_to_end(key)
        return self.orderedDict[key]

    def put(self, key: int, value: int) -> None:
        if key in self.orderedDict:
            self.orderedDict[key] = value
            self.orderedDict.move_to_end(key)
            return
        if len(self.orderedDict) >= self.capacity:
            self.orderedDict.popitem(last=False)
        self.orderedDict[key] = value
        return

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)



###### TREES ######
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 104. Maximum Depth of Binary Tree (https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) - Easy
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0

# 100. Same Tree (https://leetcode.com/problems/same-tree/description/) - Easy
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None and q == None:
            return True
        if p == None or q == None or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# 572. Subtree of Another Tree (https://leetcode.com/problems/subtree-of-another-tree/description/) - Easy
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        '''
        Time Complexity: O(n * m)
        Space Complexity: O(n + m)
        Where, `n` is the number of nodes in `root` and `m` is the number of nodes in `subRoot`.
        '''
        if self.isSametree(root, subRoot):
            return True
        if root != None:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        return False

    def isSametree(self,  root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 == None and root2 == None:
            return True
        if root1 == None or root2 == None or root1.val != root2.val:
            return False
        return self.isSametree(root1.left, root2.left) and self.isSametree(root1.right, root2.right)
# 572. Subtree of Another Tree (https://leetcode.com/problems/subtree-of-another-tree/description/) - Easy
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        return self.serialize(subRoot) in self.serialize(root)
    
    def serialize(self, node: Optional[TreeNode]) -> str:
        if node == None: return "N"
        return f"({node.val},{self.serialize(node.left)},{self.serialize(node.right)})"


# 235. Lowest Common Ancestor of a Binary Search Tree (https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        Time Complexity: O(h), where h is the height of the tree.
        Space Complexity: O(1), since we are not using any extra space.
        where, h is log(n) in a balanced tree, and n is the number of nodes in the tree.

        This is a Binary Search Tree, so we can use the properties of BST to find the LCA.
        In a BST, left child nodes are less than the parent node, and right child nodes are greater than the parent node.
        Traditionally, nodes have unique values in a BST.
        '''
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


# 236. Lowest Common Ancestor of a Binary Tree (https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        Time Complexity: O(n), where n is the number of nodes in the tree.
        Space Complexity: O(n), where n is the number of nodes in the tree.
        This is not a Binary Search Tree, just a regular binary tree, so in worst case, we have to traverse all the nodes to find the LCA.
        '''
        self.searchedLeft = set()
        self.searchedRight = set()
        targets = [p, q]
        path = self.search(root, targets, [root])
        targets = [q] if path[-1] == p else [p]
        for node in reversed(path):
            if self.search(node, targets, [node]) != None:
                return node
        return root
            
    def search(self, node: 'TreeNode', targets: List['TreeNode'], path: List['TreeNode']) -> List['TreeNode']:
        if node == None:
            return None
        path.append(node)
        if node in targets:
            return path
        left = None
        if node not in self.searchedLeft:
            self.searchedLeft.add(node)
            left = self.search(node.left, targets, path)
        if left != None:
            return left
        if node in self.searchedRight:
            return None
        self.searchedRight.add(node)
        return self.search(node.right, targets, path)


# 102. Binary Tree Level Order Traversal (https://leetcode.com/problems/binary-tree-level-order-traversal/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        levels = []
        queue = [root]
        while queue:
            level = []
            newQueue = []
            for node in queue:
                if node != None:
                    level.append(node.val)
                    newQueue.append(node.left)
                    newQueue.append(node.right)
            if level:
                levels.append(level)
            queue = newQueue
        return levels


# 98. Validate Binary Search Tree (https://leetcode.com/problems/validate-binary-search-tree/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        '''
        Time Complexity: O(n), where n is the number of nodes in the tree.
        We would check each node exactly once, so time complexity is O(n).
        '''
        return self.isValidChild(root, float('-inf'), float('inf'))
    def isValidChild(self, root: Optional[TreeNode], minNeeded: int, maxAllowed: int) -> bool:
        if root == None:
            return True
        if root.val <= minNeeded or root.val >= maxAllowed:
            return False
        return self.isValidChild(root.left, minNeeded, root.val) and self.isValidChild(root.right, root.val, maxAllowed)


# 230. Kth Smallest Element in a BST (https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        '''
        Time Complexity: O(n)
        Space Complexity: O(n)
        More precise time complexity is O(h+k), where h is height of the BST and k is given k value. Worst case h = n and k = n => O(2n) => O(n)
        1. Keep moving left until you hit none when you hit none, the previous value in the stack is the value of smallest value
        2. To find the next smallest value, go one branch to the right, and keep moving left. If the right node is none go back to parent using the stack, repeat
        '''
        node = root
        stack = [] 
        while k > 0:
            while node.left != None:
                stack.append(node)
                node = node.left
            k-=1
            if k==0:
                return node.val
            if node.right:
                node = node.right
            else:
                node = stack.pop(-1)
                node.left = None
        return node.val


# 105. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        '''
        Time Complexity: O(n^2)
        Space Complexity: O(n^2). where, n is number of nodes
        This is a sub par solution, but is easy to think of and implement. Better solution has a time complexity of O(n).
        '''
        n = len(preorder)
        index = {num: i for i, num in enumerate(inorder)}

        def subTree(preStart: int, preEnd: int, inStart: int, inEnd: int) -> Optional[TreeNode]:
            if preStart == preEnd:
                return None
            root = TreeNode(preorder[preStart])
            treeLen = index[root.val] + 1 - inStart
            root.left = subTree(preStart + 1, preStart + treeLen, inStart, inStart + treeLen - 1)
            root.right = subTree(preStart + treeLen, preEnd, inStart + treeLen, inEnd)
            return root
        
        return subTree(0, n, 0, n)

# 105. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        '''
        Time Complexity: O(n^2)
        Space Complexity: O(n^2). where, n is number of nodes, stack trace and list slicing are causes.
        This is a sub par solution, but is easy to think of and implement. Better solution has a time and space complexities of O(n).
        '''
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        leftLimit = inorder.index(root.val)+1
        root.left = self.buildTree(preorder[1:leftLimit], inorder[:leftLimit-1])
        root.right = self.buildTree(preorder[leftLimit:], inorder[leftLimit:])
        return root

# 105. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        '''
        Time Complexity: O(n^2)
        Space Complexity: O(n). where, n is number of nodes, this is because of stacktrace, we eliminated list slicing in this approach.
        This is a sub par solution, better than slicing lists though. Better solution has a time and space complexities of O(n).
        '''
        n = len(preorder)
        def subTree(preStart: int, preEnd: int, inStart: int, inEnd: int) -> Optional[TreeNode]:
            if preStart == preEnd:
                return None
            root = TreeNode(preorder[preStart])
            treeLen = inorder.index(root.val, inStart, inEnd) + 1 - inStart
            root.left = subTree(preStart + 1, preStart + treeLen, inStart, inStart + treeLen - 1)
            root.right = subTree(preStart + treeLen, preEnd, inStart + treeLen, inEnd)
            return root
        return subTree(0, n, 0, n)

# 105. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - Medium
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        '''
        Time Complexity: O(n)
        Space Complexity: O(n). where, n is number of nodes, because of stacktrace.
        This is an optimal solution, but there is even better optimized solution with same time and space complexities.
        This is good for interview purposes. Refer https://neetcode.io/solutions/construct-binary-tree-from-preorder-and-inorder-traversal
        '''
        n = len(preorder)
        index = {num: i for i, num in enumerate(inorder)}
        def subTree(preStart: int, preEnd: int, inStart: int, inEnd: int) -> Optional[TreeNode]:
            if preStart == preEnd:
                return None
            root = TreeNode(preorder[preStart])
            treeLen = index[root.val] + 1 - inStart
            root.left = subTree(preStart + 1, preStart + treeLen, inStart, inStart + treeLen - 1)
            root.right = subTree(preStart + treeLen, preEnd, inStart + treeLen, inEnd)
            return root
        return subTree(0, n, 0, n)


# 124. Binary Tree Maximum Path Sum (https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) - Hard
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxVal = float('-inf')
        self.nodeMax(root)
        return self.maxVal
    def nodeMax(self, node) -> int:
        if node == None:
            return float('-inf')
        left = self.nodeMax(node.left)
        right = self.nodeMax(node.right)
        nodeMax = max(node.val + left, node.val + right, node.val)
        self.maxVal = max(self.maxVal, nodeMax, node.val + left + right)
        return nodeMax


# 297. Serialize and Deserialize Binary Tree (https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/) - Hard
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        from json import dumps
        return dumps(self.getPreorder(root))

    def getPreorder(self, root):
        if root == None:
            return None
        return [root.val,self.getPreorder(root.left),self.getPreorder(root.right)]

    def createTree(self, array):
        if array == None:
            return None
        root = TreeNode(array[0])
        root.left = self.createTree(array[1])
        root.right = self.createTree(array[2])
        return root

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        from json import loads
        return self.createTree(loads(data))

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))



###### HEAP / PRIORITY QUEUE   ######
# 295. Find Median from Data Stream (https://leetcode.com/problems/find-median-from-data-stream/description/) - Hard
import heapq
class MedianFinder:

    def __init__(self):
        self.left = []
        self.right = []

    def addNum(self, num: int) -> None:
        if self.left == [] or num <= -self.left[0]:
            heapq.heappush(self.left, -num)
        else:
            heapq.heappush(self.right, num)

        if len(self.left) > len(self.right) + 1:
            heapq.heappush(self.right, -heapq.heappop(self.left))
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))

    def findMedian(self) -> float:
        if len(self.left) == len(self.right):
            return (-self.left[0] + self.right[0]) / 2.0
        return -self.left[0]

# Example usage:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()




###### TRIES ######
# 212. Word Search II (https://leetcode.com/problems/word-search-ii/description/) - Hard
class Trie:
    def __init__(self):
        self.children = {}
        self.isWord = False

    def add(self, word: str):
        cur = self
        for i, ch in enumerate(word):
            if ch not in cur.children: cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isWord = True

    def remove(self, word: str):
        cur = self
        nodes = []
        for i, ch in enumerate(word):
            if ch not in cur.children: return
            nodes.append(cur)
            cur = cur.children[ch]
        cur.isWord = False
        i = -1
        while nodes:
            parent = nodes.pop()
            if len(cur.children) > 0 or cur.isWord: return
            parent.children.pop(word[i])
            cur = parent
            i -= 1

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
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

# 212. Word Search II (https://leetcode.com/problems/word-search-ii/description/) - Hard
class PrefixTree:
    '''
    Same thing not much difference, just added same pruning, didn't make much difference, Over engineered!
    '''
    def __init__(self) -> None:
        self.children = {}
        self.isWord = None
    
    def add(self, word: str) -> None:
        root = self
        for ch in word:
            if ch not in root.children:
                root.children[ch] = PrefixTree()
            root = root.children[ch]
        root.isWord = word

    def remove(self, word: str) -> str:
        if not word: # Not needed
            return None
        root = self
        deleteLink = (root, word[0])
        for ch in word:
            if len(root.children) > 1 or root.isWord:
                deleteLink = (root, ch)
            if ch not in root.children:
                return None
            root = root.children[ch]
        if not root.children:
            deleteLink[0].children.pop(deleteLink[1])
        root.isWord = None
        return word
    
    def __str__(self) -> str:
        return f'Node:(children={self.children.keys()}, isWord={self.isWord})'

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # Prune words
        words = set(words)
        boardCounter = sum((Counter(row) for row in board),Counter())
        self.allWords = dict()
        while words:
            word = words.pop()
            if len(word) > len(board)*len(board[0]):
                continue
            wordCounter = Counter(word)
            for ch in wordCounter:
                if wordCounter[ch] > boardCounter[ch]:
                    continue

            # Optimize branching (doesn't contribute much), ideal would be to see if there's any algo for word commanlity, reduce number of prefix branches..
            reverseWord = word[::-1]
            key = word
            if boardCounter[word[0]] > boardCounter[word[-1]]:
                key = reverseWord
            
            # Reduce Prefix tree branches by reducing number of words to search
            if reverseWord in words:
                words.remove(reverseWord)
                self.allWords[key] = [word, reverseWord]
            else:
                self.allWords[key] = [word]

        self.visited = set()
        for r in range(len(board)):
            self.visited.update([(r,-1), (r,len(board[0]))])
        for c in range(len(board[0])):
            self.visited.update([(-1,c), (len(board),c)])
        
        self.prefixTree = PrefixTree()
        for word in self.allWords:
            self.prefixTree.add(word)

        self.board = board
        self.output = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                self.search(r,c,self.prefixTree)
        return self.output

    def search(self, r: int, c: int, node: PrefixTree) -> None:
        if not node: # Not needed
            return
        
        if node.isWord:
            self.output.extend(self.allWords.pop(node.isWord, []))
            self.prefixTree.remove(node.isWord)
        
        if (r,c) in self.visited or self.board[r][c] not in node.children:
            return
        
        node = node.children[self.board[r][c]]
        self.visited.add((r,c))
        self.search(r-1, c, node)
        self.search(r+1, c, node)
        self.search(r, c-1, node)
        self.search(r, c+1, node)
        self.visited.remove((r,c))
        return



###### GRAPHS ######
# 417. Pacific Atlantic Water Flow (https://leetcode.com/problems/pacific-atlantic-water-flow/description/) - Medium
class Solution:
    '''
    Time Complexity: O(R * C)
    Space Complexity: O(R * C)
    Where, R is number of rows, C is number of columns in heights.

    Note:
    Optimal Solution:
    The best approach is to do DFS of ocean border cells only, and find cells that can reach that ocean.
    Basically solving in reverse, going from ocean to their respective reachable cells, and find cells common in both oceans.

    My Mistakes:
    Instead for some reason, when you're trying to code it, you're some how coding the brute force approach, 
    where you're doing DFS for every cell, to see if it can reach both oceans with memoization.
    This approach is way more complex to code, and has worse time and space complexities of O((R * C)^2).
    REMEMBER TO PERFORM DFS FOR OCEAN BORDER CELLS ONLY!!! Not for every cell, 
    and you don't need a visited set, you're already using pacific and atlantic sets 
    that act as visited sets for respective oceans.
    '''
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        R = len(heights)
        C = len(heights[0])

        pacific = set()
        atlantic = set()

        def dfs(r,c,ocean,prevH):
            if not ( 0 <= r < R and 0 <= c < C ) or heights[r][c] < prevH:
                return False
            if (r,c) in ocean:
                return True
            
            ocean.add((r,c)) # This prevents revisiting the same cell
            dfs(r+1,c,ocean,heights[r][c]) 
            dfs(r-1,c,ocean,heights[r][c]) 
            dfs(r,c+1,ocean,heights[r][c]) 
            dfs(r,c-1,ocean,heights[r][c])
            return True
    
        # Only doing DFS for ocean border cells, NOT FOR EVERY CELL!!!
        for c in range(C):
            dfs(0,c,pacific,heights[0][c])
            dfs(R-1,c,atlantic,heights[R-1][c])
        
        for r in range(R):
            dfs(r,0,pacific,heights[r][0])
            dfs(r,C-1,atlantic,heights[r][C-1])

        return list(pacific & atlantic)


# 207. Course Schedule  (https://leetcode.com/problems/course-schedule/description/) - Medium
class Solution:
    '''
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    Where, V is number of courses, E is number of prerequisites. 
    We're only doing dfs for each course once, and checking each prerequisite link once, because we're using memoization.
    '''
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        courseDependencies = dict()
        possibleCourses = set()

        for course, dependency in prerequisites:
            courseDependencies.setdefault(course, set()).add(dependency)
            courseDependencies.setdefault(dependency, set())
        
        visited = set()

        def isCoursePossible(course):
            if course in visited:
                return False
            if course in possibleCourses:
                return True

            visited.add(course)
            
            for dependency in courseDependencies[course]:
                if not isCoursePossible(dependency):
                    return False
            
            visited.discard(course)
            possibleCourses.add(course)
            return True

        for course in courseDependencies:
            if not isCoursePossible(course):
                return False
        
        return True


# 261. Graph Valid Tree (https://leetcode.com/problems/graph-valid-tree/description/) - Medium - Premium (https://neetcode.io/problems/valid-tree/question)
class Solution:
    '''
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    Where, V is number of nodes, E is number of edges. 
    We're only doing dfs for each node once, and checking each edge once, because we're using visited set.
    1. A valid tree should have exactly n-1 edges, if there are more edges, there must be a cycle.
    2. A valid tree should be fully connected, meaning all nodes should be reachable from any node. We can check this by doing a DFS/BFS from any node and see if we can visit all nodes.
    3. During DFS/BFS, if we encounter a visited node that is not the parent of the current node, then there is a cycle.
    4. Finally, after DFS/BFS, if the number of visited nodes is not equal to n, then the graph is not fully connected.
    '''
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) > (n - 1):
            return False

        hmap = dict()

        for n1, n2 in edges:
            hmap.setdefault(n1, set()).add(n2)
            hmap.setdefault(n2, set()).add(n1)
        
        visited = set()

        def dfs(n1, prev):
            if n1 in visited:
                return False
            visited.add(n1)
            for x in hmap.get(n1, set()): # Provide a default set is necessary cause if there is only 1 node, it won't have edges, it wouldn't have been saved in `hmap`
                if x != prev and not dfs(x, n1):
                    return False            
            return True

        return dfs(0,-1) and len(visited) == n


# 323. Number of Connected Components in an Undirected Graph (https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/) - Medium - Premium (https://neetcode.io/problems/count-connected-components/question)
class Solution:
    '''
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    Where, V is number of nodes, E is number of edges.
    We're only doing dfs for each node once, and checking each edge once, because we're using visited set.

    There is a more efficient solution using union-find, whose time complexity is also O(V + E),
    but this solution is easier to understand and implement.
    '''
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        hmap = dict()
        visited = set()
        count = 0

        for n1, n2 in edges:
            hmap.setdefault(n1, set()).add(n2)
            hmap.setdefault(n2, set()).add(n1)

        def dfs(x):
            if x in visited:
                return
            visited.add(x)
            for k in hmap.get(x, set()):
                dfs(k)
            return

        for x in range(n):
            if x not in visited:
                dfs(x)
                count+=1
        
        return count



###### ADVANCED GRAPHS ######

# 269. Alien Dictionary (https://leetcode.com/problems/alien-dictionary/description/) - Hard - Premium (https://neetcode.io/problems/foreign-dictionary/question)
class Solution:
    '''
    Time complexity: O(N+V+E)
    Space complexity: O(V+E)

    Where V is the number of unique characters, E is the number of edges, 
    and N is the total number of characters in all the words.

    This is a hard problem, we used topological sort to solve this problem with DFS,
    can also be solved with BFS using Kahn's algorithm (another Topological sort).
    We used a graph to represent the characters and their order, (we used an adjacency list for this),
    and we used topological sort to find the order of characters.
    1. We first create a graph where each character points to the characters that come after it.
    2. We then do a topological sort on the graph, and if we encounter a cycle, 
       we return an empty string. 
       (Cycles mean, there is no valid order for characters, 
       For Eg: We already have A < B, B < C, and if we get C < A which is not possible)
    3. If we don't encounter a cycle, we return the order of characters.
    4. When we are doing topological sort, we process the characters in reverse order, 
       basically we process all the children, and when it's a leaf node (no children for this node) process it, 
       and backtrack to the parent nodes, that's why the output we append ends up coming in reverse order, 
       which we reverse in the end.
    '''
    def foreignDictionary(self, words: List[str]) -> str:
        hmap = {}
        
        for w in words:
            for c in w:
                hmap[c] = set()
        
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            l = min(len(w1), len(w2))
            if w1[:l] == w2[:l] and len(w1) > len(w2):
                return ""
            for j in range(l):
                if w1[j] != w2[j]:
                    hmap[w1[j]].add(w2[j])
                    break # Breaking here is crucial, 
                          # Eg: ABCZY < ABDE, here C < D and if you won't break here, 
                          # you'll end up adding Z < E which is not true!
        
        visiting = set()
        processed = set()

        output = []

        def dfs(c):
            if c in visiting:
                return False
            if c in processed:
                return True
            
            visiting.add(c)

            for k in hmap[c]:
                if not dfs(k):
                    return False

            visiting.discard(c)
            output.append(c)
            processed.add(c)
            return True

        for c in hmap:
            if not dfs(c):
                return ""

        return "".join(reversed(output))



###### 1-D DYNAMIC PROGRAMMING ######

# 70. Climbing Stairs (https://leetcode.com/problems/climbing-stairs/description/) - Easy
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        Time Complexity: O(n)
        Space Complexity: O(n)
        Where, n is number of stairs.
        Constrains specified as 1 <= n <= 45.

        You were going with `1 + dfs(n-1) + dfs(n-2)` to compensate for n=0 case, 
        but that causes over-counting, if you look at constrains, n >= 1, 
        so you can go with right logic `dfs(n-1) + dfs(n-2)` and `if n==0 return 1`, 
        you don't need to do `+1`, and be sure you return 1 for `n == 0` cases.
        '''
        cache = {}
        def dfs(n):
            if n < 2:
                return 1
            if n in cache:
                return cache[n]
            res = dfs(n-1) + dfs(n-2)
            cache[n] = res
            return res
        return dfs(n)


# 198. House Robber (https://leetcode.com/problems/house-robber/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of houses.

    Simple DFS with memoization problem. But the most optimal solution is below, 
    has O(1) space complexity.
    '''
    def rob(self, nums: List[int]) -> int:
        cache = {}

        def dfs(i):
            if i in cache:
                return cache[i]
            if i >= len(nums):
                return 0
            res = max(nums[i] + dfs(i+2), dfs(i+1))
            cache[i] = res
            return res
        
        return dfs(0)

# 198. House Robber (https://leetcode.com/problems/house-robber/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of houses.
    This is the most optimal solution with O(1) space complexity.
    '''
    def rob(self, nums: List[int]) -> int:
        rob1, rob2 = 0, 0

        for house in nums:
            temp = rob2
            rob2 = max(house + rob1, rob2)
            rob1 = temp

        return rob2


# 213. House Robber II (https://leetcode.com/problems/house-robber-ii/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of houses.
    Most optimal solution has space complexity of O(1), similar to House Robber I problem.
    Since the houses are in a circle, we can't rob the first and last house together.
    '''
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        cache = {}
        def dfs(i, leaveLast):
            if (i,leaveLast) in cache:
                return cache[(i,leaveLast)]
            if (leaveLast and i >= len(nums) - 1) or i >= len(nums):
                return 0
            res = max(nums[i] + dfs(i+2, leaveLast), dfs(i+1, leaveLast))
            cache[(i,leaveLast)] = res
            return res
            
        return max(dfs(0, True), dfs(1, False))

# 213. House Robber II (https://leetcode.com/problems/house-robber-ii/description/) - Medium - Duplicate
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of houses.
    This is the most optimal solution, has space complexity of O(1), similar to House Robber I problem.
    Since the houses are in a circle, we can't rob the first and last house together.
    '''
    def rob(self, nums: List[int]) -> int:
        def helper(houses):
            rob1, rob2 = 0, 0
            for house in houses:
                newRob = max(rob1 + house, rob2)
                rob1 = rob2
                rob2 = newRob
            return rob2
        return max(nums[0], helper(nums[1:]), helper(nums[:-1]))


# 5. Longest Palindromic Substring (https://leetcode.com/problems/longest-palindromic-substring/description/) - Medium
class Solution:
    def longestPalindrome(self, s: str) -> str:
        '''
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        Where, n is the length of the given string.
        
        NOTE: This is not the most optimal solution, we used Two pointers approach which is good enough for interview purposes.
        Can also be done with Dynamic Programming with O(n^2) time and space complexity.
        Most optimal solution is Manacher's Algorithm with time complexity of O(n) and space complexity of O(n).
        Refer https://neetcode.io/problems/longest-palindromic-substring/solution for most optimal solution.
        '''
        left, right = 0, 0
        maxLeft, maxRight = 0, 0

        def updateMax():
            nonlocal left, right, maxLeft, maxRight
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left-=1
                right+=1
            left+=1
            right-=1
            if right-left > maxRight-maxLeft:
                maxLeft = left
                maxRight = right

        for i in range(len(s)-1):
            left = i
            right = i
            updateMax()    
            left = i
            right = i+1
            updateMax()
        
        return s[maxLeft:maxRight+1]


# 647. Palindromic Substrings (https://leetcode.com/problems/palindromic-substrings/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    Where, n is the length of the given string.
    NOTE: This is not the most optimal solution, we used Two pointers approach which is good enough for interview purposes.
    Can also be done with Dynamic Programming with O(n^2) time and space complexity.
    Most optimal solution is Manacher's Algorithm with time complexity of O(n) and space complexity of O(n).
    Refer https://neetcode.io/problems/palindromic-substrings/solution for most optimal solution.
    '''
    def countSubstrings(self, s: str) -> int:
        left, right = 0, 0
        count = 0

        def countPalindromes():
            nonlocal left, right, count
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
                count += 1

        for i in range(len(s)-1):
            left = i
            right = i
            countPalindromes()
            left = i
            right = i+1
            countPalindromes()

        return count+1


# 91. Decode Ways (https://leetcode.com/problems/decode-ways/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is the length of the given string.
    Simple DFS with memoization problem.
    '''
    def numDecodings(self, s: str) -> int:
        # cache = {}

        @cache # Using inbuilt cache decorator, code to import: `from functools import cache`
        def dfs(i):
            # if i in cache:
            #     return cache[i]
            if i >= len(s):
                return 1
            res = 0
            if s[i] != '0':
                res += dfs(i+1)
            if i < len(s)-1 and (s[i] == '1' or (s[i] == '2' and s[i+1] in '0123456')):
                res += dfs(i+2)
            # cache[i] = res
            return res    
        
        return dfs(0)


# 322. Coin Change (https://leetcode.com/problems/coin-change/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n * m)
    Space Complexity: O(n)
    Where, n is the amount, m is number of coins.
    This is a Top-Down approach, a recursive solution with memoization.
    This is more intuitive to think of than Bottom-Up approach.
    '''
    def coinChange(self, coins: List[int], amount: int) -> int:
        cache = {0: 0}

        def dfs(target):
            if target in cache:
                return cache[target]
            if target < 0:
                return -1

            res = float('inf') # Can use `amount + 1` instead of `float('inf')` because max coins needed will be amount (all 1s) if "1" coin exists
            for coin in coins:
                tres = dfs(target-coin)
                if tres > -1:
                    res = min(res, tres+1)

            if res == float('inf'): # Can use `amount + 1` instead of `float('inf')`
                cache[target] = -1
                return -1 
            
            cache[target] = res
            return res

        return dfs(amount)

# 322. Coin Change (https://leetcode.com/problems/coin-change/description/) - Medium - Duplicate
class Solution:
    '''
    Time Complexity: O(n * m)
    Space Complexity: O(n)
    Where, n is the amount, m is number of coins.
    This is a Bottom-Up approach, an iterative solution with Dynamic Programming.
    '''
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])
        return dp[amount] if dp[amount] != amount + 1 else -1


# 152. Maximum Product Subarray (https://leetcode.com/problems/maximum-product-subarray/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of elements in the given array.
    Simple DFS with memoization problem, the most optimal solution has O(1) space complexity.
    We need to keep track of both maximum and minimum products at each step, because a negative number can turn a minimum product into a maximum product.
    '''
    def maxProduct(self, nums: List[int]) -> int:
        maxV = nums[-1] # Since at least one number exists, initializing with last number, if we initialize with first number, you'll need to handle edge case where last number is the maximum product. Eg: [-4,-1]. Instead of just returning 
        
        @cache
        def dfs(i):
            if i == len(nums)-1:
                return nums[i], nums[i]
            nonlocal maxV
            pos, neg = dfs(i+1)
            resPos = max(nums[i], nums[i] * pos, nums[i] * neg)
            resNeg = min(nums[i], nums[i] * pos, nums[i] * neg)
            maxV = max(resPos, maxV)
            return resPos, resNeg

        dfs(0)
        return maxV

# 152. Maximum Product Subarray (https://leetcode.com/problems/maximum-product-subarray/description/) - Medium - Duplicate
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of elements in the given array.
    This is the most optimal solution with O(1) space complexity.
    This is a Kadane's Algorithm variation.
    Can also be done using Prefix and Suffix product approach.
    Look in https://neetcode.io/problems/maximum-product-subarray/solution'''
    def maxProduct(self, nums: List[int]) -> int:
        res = nums[0]
        curMin, curMax = 1, 1

        for num in nums:
            tmp = curMax * num
            curMax = max(num * curMax, num * curMin, num)
            curMin = min(tmp, num * curMin, num)
            res = max(res, curMax)
        return res


# 139. Word Break (https://leetcode.com/problems/word-break/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n * m * k)
    Space Complexity: O(n)
    Where, n is length of the string, m is number of words in the dictionary, k is average length of the words in the dictionary.'''
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        @cache # Using inbuilt cache decorator, code to import: `from functools import cache`
        def dfs(i):
            if i >= len(s):
                return True
            for word in wordDict:
                if s[i:i+len(word)] == word and dfs(i+len(word)):
                    return True
            return False

        return dfs(0)

# 139. Word Break (https://leetcode.com/problems/word-break/description/) - Medium - Duplicate
class Solution:
    '''
    Time Complexity: O(n * m * k)
    Space Complexity: O(n + m)
    Where, n is length of the string, m is number of words in the dictionary, k is average length of the words in the dictionary.
    Small optimization (kind of insignificant) for larger wordDicts, by precomputing lengths of words in the dictionary.
    '''
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        lengths = set([len(word) for word in wordDict])

        @cache # Using inbuilt cache decorator, code to import: `from functools import cache`
        def dfs(i):
            if i >= len(s):
                return True
            for length in lengths:
                if s[i:i+length] in wordDict and dfs(i+length):
                    return True
            return False

        return dfs(0)


# 300. Longest Increasing Subsequence (https://leetcode.com/problems/longest-increasing-subsequence/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    Where, n is number of elements in the given array.
    This is not the most optimal solution, but good enough for interview. Most optimal one has time complexity of O(n log n).
    1. Keep a cache of longest increasing subsequence for each prev number.
    2. For each number, we check list of all previous numbers, if the current number is greater than the previous number,
       we can extend the increasing subsequence for it by 1.
    3. We update the cache for the current number with the maximum length found.
    4. Finally, we return the maximum length from the cache.
    Eg: [10,2,5,3,7,18], when current number is 7, 
       previous numbers in cache are {10:1, 2:1, 5:2, 3:2}, 
       we can extend the sequence from either 2, 5 or 3, since they are all less than 7,
       but the previous maximum subsequence length is 2 (from either 5 or 3), 
       so we choose one of 5 or 3 and extend it to length 3, by considering 7 as the next number in the sequence.
       We update the cache for 7 as 3, and also update the maximum length if needed.
    '''
    def lengthOfLIS(self, nums: List[int]) -> int:
        cache = {}
        maxLen = 1
        for num in nums:
            val = 1
            for prev in cache:
                if num > prev:
                    val = max(val, 1 + cache[prev])
            cache[num] = val
            maxLen = max(maxLen, val)
        return maxLen

# 300. Longest Increasing Subsequence (https://leetcode.com/problems/longest-increasing-subsequence/description/) - Medium - Duplicate
from bisect import bisect_left
class Solution:
    '''
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Where, n is number of elements in the given array.
    This is the most optimal solution using Binary Search. 
    Watch this video for intuition: (https://youtu.be/on2hvxBXJH4)
    NOTE: The dummyList is only for maintaining the length of the longest increasing subsequence found so far,
    and not for storing the actual subsequence, in fact if you print the dummyList, it will not be the actual subsequence.
    '''
    def lengthOfLIS(self, nums: List[int]) -> int:
        dummyList = []
        for num in nums:
            i = bisect_left(dummyList,num) # `from bisect import bisect_left`
            if i == len(dummyList):
                dummyList.append(num)
            else:
                dummyList[i] = num
        return len(dummyList)



###### 2-D DYNAMIC PROGRAMMING ######

# 62. Unique Paths (https://leetcode.com/problems/unique-paths/description/) - Medium
class Solution:
    '''
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    Where, m is number of rows, n is number of columns in the grid.
    '''
    def uniquePaths(self, m: int, n: int) -> int:
        @cache
        def dp(i,j):
            if i >= m or j >= n:
                return 0
            if i == m-1 or j == n-1:
                return 1
            return dp(i+1,j) + dp(i,j+1)
        return dp(0,0)


# 1143. Longest Common Subsequence (https://leetcode.com/problems/longest-common-subsequence/description/) - Medium
class Solution:
    '''
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    Where, m is length of text1, n is length of text2.
    There is also a space optimized solution with O(min(m, n)) space complexity, 
    but this is good enough for interview.
    '''
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        @cache
        def dp(i, j):
            if i >= len(text1) or j >= len(text2):
                return 0
            if text1[i] == text2[j]:
                return 1 + dp(i+1,j+1)
            else:
                return max(dp(i, j+1), dp(i+1, j))
        return dp(0,0)



###### BACKTRACKING ######
# 39. Combination Sum (https://leetcode.com/problems/combination-sum/description/) - Medium
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        self.candidates = candidates
        return self.getCombination(0, target, [])

    def getCombination(self, i, need, stack):
        if need == 0:
            return [stack]
        output = []
        while i < len(self.candidates) and self.candidates[i] <= need:
            output += self.getCombination(i, need - self.candidates[i], stack + [self.candidates[i]])
            i += 1
        return output


# 79. Word Search (https://leetcode.com/problems/word-search/description/) - Medium
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        '''
        Time Complexity: O(R * C * 4^L)
        Space Complexity: O(L), for stack of visited cells
        Where, R is number of rows, C is number of columns in board, and L is number of characters in word.
        '''
        self.board = board
        self.word = word
        self.visited = set()
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.isWord(r, c, 0):
                    return True
        return False

    def isWord(self, r: int, c: int, i: int) -> bool:
        if i >= len(self.word):
            return True
        if (r < 0 or c < 0 or r >= len(self.board) or c >= len(self.board[0])
            or (r,c) in self.visited or self.board[r][c] != self.word[i]):
            return False
        self.visited.add((r,c))
        if (self.isWord(r+1,c,i+1) or self.isWord(r-1,c,i+1)
            or self.isWord(r,c+1,i+1) or self.isWord(r,c-1,i+1)):
            return True
        self.visited.remove((r,c))

# 79. Word Search (https://leetcode.com/problems/word-search/description/) - Medium
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        '''
        Although the time and space complexities are not changed, this is a quicker solution in real world because of inexpensive preliminary pruning.
        Time Complexity: O(R * C * 4^L)
        Space Complexity: O(L), for stack of visited cells
        Where, R is number of rows, C is number of columns in board, and L is number of characters in word.
        '''
        # There won't be a solution if length of the word is bigger than number of characters in board
        if len(word) > len(board) * len(board[0]):
            return False

        # There won't be a solution if frequency for any of the character in a word, is more than the frequency of that character on board
        wordCount = Counter(word)
        boardCount = sum((Counter(row) for row in board), Counter())
        for ch in wordCount:
            if boardCount[ch] < wordCount[ch]:
                return False

        # We can reduce the amount of branching significantly by picking the tip of the word with least number of repeats on board
        if boardCount[word[0]] > boardCount[word[-1]]:
            word = word[::-1]

        self.board = board
        self.word = word
        self.visited = set()
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.isWord(r, c, 0):
                    return True
        return False

    def isWord(self, r: int, c: int, i: int) -> bool:
        if i >= len(self.word):
            return True
        if (r < 0 or c < 0 or r >= len(self.board) or c >= len(self.board[0])
            or (r,c) in self.visited or self.board[r][c] != self.word[i]):
            return False
        self.visited.add((r,c))
        if (self.isWord(r+1,c,i+1) or self.isWord(r-1,c,i+1)
            or self.isWord(r,c+1,i+1) or self.isWord(r,c-1,i+1)):
            return True
        self.visited.remove((r,c))


# 78. Subsets (https://leetcode.com/problems/subsets/description/) - Medium
from copy import deepcopy
class Solution:
    '''
    NOT THE  MOST EFFICIENT SOLUTION
    '''
    def subsets(self, nums: List[int]) -> List[List[int]]:
        output = [[]]
        for n in nums:
            l = len(output)
            output = output + deepcopy(output)
            for i in range(l):
                output[i].append(n)
        return output


######  GREEDY  ######
# 53. Maximum Subarray (https://leetcode.com/problems/maximum-subarray/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of elements in the given array.
    This is Kadane's Algorithm, this is the most efficient solution for this problem.
    '''
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = -100000
        curSum = 0
        for n in nums:
            curSum += n
            maxSum = max(maxSum, curSum)
            if curSum < 0:
                curSum = 0
        return maxSum


# 55. Jump Game (https://leetcode.com/problems/jump-game/description/) - Medium
class Solution:
    '''
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    Where, n is number of elements in the given array.
    This is not a the most efficient solution.
    '''
    def canJump(self, nums: List[int]) -> bool:
        @cache
        def dfs(i):
            if i >= len(nums) - 1:
                return True
            for step in range(1, nums[i]+1):
                if dfs(i+step):
                    return True
            return False
        return dfs(0)
    
# 55. Jump Game (https://leetcode.com/problems/jump-game/description/) - Medium - Duplicate
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of elements in the given array.
    This is the most efficient solution.
    '''
    def canJump(self, nums: List[int]) -> bool:
        i = len(nums) - 1
        possible = i
        while i >= 0:
            if i + nums[i] >= possible:
                possible = i
            i -= 1
        return possible == 0




#########  EXTRA PROBLEMS  #########

# 3365. Rearrange K Substrings to Form Target String (https://leetcode.com/problems/rearrange-k-substrings-to-form-target-string/description/) - Medium
class Solution:
    def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
        eqLen = len(s)//k
        sCounter = defaultdict(int)
        for i in range(0, len(s), eqLen):
            sCounter[s[i:i+eqLen]] += 1
            
        for i in range(0, len(t), eqLen):
            part = t[i:i+eqLen]
            if sCounter[part] < 1:
                return False
            sCounter[part] -= 1
        return True
                

# 3366. Minimum Array Sum (https://leetcode.com/problems/minimum-array-sum/description/) - Medium
# NOT THE  MOST EFFICIENT SOLUTION
class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        '''
        Works flawlessly! This is Dynamic Programming approach, but Greedy algorithm is also possible, and is the best implementation (Implementing greedy algo is little hard, to figure out the corner case)!
        Time Complexity: O(n * op1 * op2)
        Space Complexity: O(n * op1 * op2)
        '''
        memo = dict()

        def apply(i: int, op1: int, op2: int) -> int:
            if i >= len(nums): return 0
            if (i, op1, op2) in memo: return memo[(i, op1, op2)]

            n = nums[i]
            answer = n + apply(i+1, op1, op2)
            if op1: answer = min(answer, (n+1)//2 + apply(i+1, op1-1, op2))
            if op2 and n >= k: answer = min(answer, n-k + apply(i+1, op1, op2-1))
            if op1 and op2:
                if n >= k: answer = min(answer, (n-k+1)//2 + apply(i+1, op1-1, op2-1))
                if n >= 2*k-1: answer = min(answer, (n+1)//2 - k + apply(i+1, op1-1, op2-1))
            memo[(i, op1, op2)] = answer
            return answer

        return apply(0, op1, op2)




#######  DAILY CHALLENGES  #######

# 1161. Maximum Level Sum of a Binary Tree (https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/) - Medium - 2026-01-06
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(w)
    Where, n is number of nodes in the tree, w is maximum width of the tree.
    Simple BFS level order traversal problem.
    '''
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        maxValue = root.val
        maxLevel = 1
        level = 1
        nodes = [root]

        while nodes:
            childNodes = []
            value = 0
            for node in nodes:
                if node.left:
                    childNodes.append(node.left)
                if node.right:
                    childNodes.append(node.right)
                value += node.val
            if value > maxValue:
                maxLevel = level
                maxValue = value
            nodes = childNodes
            level += 1
        
        return maxLevel


# 1339. Maximum Product of Splitted Binary Tree (https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description/) - Medium - 2026-01-07
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of nodes in the tree.
    Although this is an optimal solution theoretically, in real world there is a more optimal solution where caching is not needed for getSum function.
    When trees is too large, cache look ups might take O(n) time instead of O(1) due to hash collisions, hence in real world this is slower.
    There is a more optimal solution where caching is not needed for getSum function, refer next solution.'''
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        maxVal = 0
        
        @cache # from functools import cache
        def getSum(node):
            if not node:
                return 0
            return node.val + getSum(node.left) + getSum(node.right)
            
        totalSum = getSum(root)

        def dfs(node):
            if not node:
                return
            nonlocal maxVal
            curSum = node.val + getSum(node.left) + getSum(node.right)
            maxVal = max(maxVal, curSum * (totalSum - curSum))
            dfs(node.left)
            dfs(node.right)
            return
        
        dfs(root)
        return maxVal % (10**9 + 7)

# 1339. Maximum Product of Splitted Binary Tree (https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description/) - Medium - 2026-01-07 - Duplicate
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of nodes in the tree.
    A cleaner solution than previous one.'''
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        maxVal = 0
        subtreeSums = list()

        def getSum(node):
            if node is None:
                return 0
            total = node.val + getSum(node.left) + getSum(node.right)
            nonlocal subtreeSums
            subtreeSums.append(total)
            return total

        totalSum = getSum(root)

        for val in subtreeSums:
            maxVal = max(maxVal, val * (totalSum - val))

        return maxVal % (10**9 + 7)


# 1458. Max Dot Product of Two Subsequences (https://leetcode.com/problems/max-dot-product-of-two-subsequences/description/) - Hard - 2026-01-08
class Solution:
    '''
    Time Complexity: O(n * m)
    Space Complexity: O(n * m)
    Where, n is length of nums1, m is length of nums2.
    Dynamic Programming with memoization problem.
    '''
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dp(i,j):
            if i >= len(nums1) or j >= len(nums2):
                return float('-inf')
            prod = nums1[i]*nums2[j]
            return max(prod, prod + dp(i+1,j+1), dp(i, j+1), dp(i+1, j))
        return dp(0,0)


# 865. Smallest Subtree with all the Deepest Nodes (https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/description/) - Medium - 2026-01-09
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(n)
    Where, n is number of nodes in the tree.
    This is good enough solution for interview purposes.
    We used BFS to find the deepest nodes, then we backtrack using parent pointers 
    to find the lowest common ancestor (LCA) of those deepest nodes. Total 2 passes are done on the tree.
    The most optimal solution uses DFS in a single pass to find the LCA of deepest nodes.
    Refer (https://youtu.be/bMXHK-ASQV0) for most optimal solution.'''
    def subtreeWithAllDeepest(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # if not root:
        #     return None
        level = [root]
        deepestNodes = level
        parent = dict()
        
        while len(level) > 0:
            newLevel = []
            for node in level:
                if node.left:
                    newLevel.append(node.left)
                    parent[node.left] = node
                if node.right:
                    newLevel.append(node.right)
                    parent[node.right] = node
            deepestNodes = level
            level = newLevel
        
        lca = set(deepestNodes)
        lca.discard(None)
        
        while len(lca) > 1:
            lca = set([parent[node] for node in lca])

        return lca.pop() # root if len(lca) == 0 else lca.pop()


# 712. Minimum ASCII Delete Sum for Two Strings (https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/description/) - Medium - 2026-01-10
class Solution:
    '''
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    Where, m is length of s1, n is length of s2.
    Dynamic Programming with memoization problem.
    '''
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        @cache        
        def dp(i,j):
            if i >= len(s1):
                return sum([ord(c) for c in s2[j:]])
            if j >= len(s2):
                return sum([ord(c) for c in s1[i:]])
            if s1[i] == s2[j]:
                return dp(i+1,j+1)
            return min(ord(s1[i]) + dp(i+1,j), ord(s2[j]) + dp(i,j+1))
        return dp(0,0)


# 85. Maximal Rectangle (https://leetcode.com/problems/maximal-rectangle/description/) - Medium - 2026-01-11
class Solution:
    '''
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    Where, m is number of rows, and n is number of columns of matrix
    DID NOT GO THROUGH THE SOLUTION YET
    '''
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        R, C = len(matrix), len(matrix[0])
        heights = [0] * C
        maxA = 0

        def largestRectangleArea(heights):
            stack = []
            res = 0
            heights.append(0)  # sentinel

            for i, h in enumerate(heights):
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    res = max(res, height * width)
                stack.append(i)

            heights.pop()
            return res

        for i in range(R):
            for j in range(C):
                if matrix[i][j] == "1":
                    heights[j] += 1
                else:
                    heights[j] = 0

            maxA = max(maxA, largestRectangleArea(heights))

        return maxA



# 1266. Minimum Time Visiting All Points (https://leetcode.com/problems/minimum-time-visiting-all-points/description/) - Easy - 2026-01-12
class Solution:
    '''
    Time Complexity: O(n)
    Space Complexity: O(1)
    Where, n is number of points.
    Simple Greedy problem.
    '''
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        time = 0
        for i in range(1, len(points)):
            hr = abs(points[i][0] - points[i-1][0])
            ve = abs(points[i][1] - points[i-1][1])
            time += max(hr, ve)
        return time


# 3453. Separate Squares I (https://leetcode.com/problems/separate-squares-i/description/) - Medium - 2026-01-13
class Solution:
    '''
    Time Complexity: O(n log m)
    Space Complexity: O(1)
    Where, n is number of squares, m is the Y co-ordinates range of the squares.
    More optimal solution uses sweep-line / prefix-area approach, but this is good enough for interview.
    '''
    def separateSquares(self, squares: List[List[int]]) -> float:
        target = sum(l*l for x,y,l in squares)/2.0
        low = min(y for x,y,l in squares)
        high = max(y+l for x,y,l in squares)
        
        def getLowArea(h):
            area = 0
            for x,y,l in squares:
                if y+l <= h:
                    area += l*l
                elif y < h < y + l:
                    area += l * (h-y)
            return area

        mid = low
        while high - low > 1e-6:
            mid = (low + high)/2.0
            area = getLowArea(mid)
            if area >= target:
                high = mid
            else:
                low = mid
        
        return mid


# 3454. Separate Squares II (https://leetcode.com/problems/separate-squares-ii/description/) - Hard - 2026-01-14
class Solution:
    '''
    Time Complexity: O((n^2) * log n)
    Space Complexity: O(n)
    Where, n is number of squares.
    Sweep-line / prefix-area approach problem.
    DID NOT GO THROUGH THE SOLUTION YET!
    '''
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Step 1: build y-events
        events = defaultdict(list)
        for x, y, l in squares:
            events[y].append((x, x + l, 1))      # add interval
            events[y + l].append((x, x + l, -1)) # remove interval

        ys = sorted(events.keys())

        # Helper to compute union length of x-intervals
        def union_length(intervals):
            if not intervals:
                return 0
            intervals.sort()
            total = 0
            cur_l, cur_r = intervals[0]
            for l, r in intervals[1:]:
                if l > cur_r:
                    total += cur_r - cur_l
                    cur_l, cur_r = l, r
                else:
                    cur_r = max(cur_r, r)
            total += cur_r - cur_l
            return total

        # Step 2: First sweep — compute total union area
        active = []
        total_area = 0.0

        for i in range(len(ys) - 1):
            y = ys[i]
            y2 = ys[i + 1]

            for x1, x2, typ in events[y]:
                if typ == 1:
                    active.append((x1, x2))
                else:
                    active.remove((x1, x2))

            width = union_length(active)
            total_area += width * (y2 - y)

        target = total_area / 2.0

        # Step 3: Second sweep — find minimum y
        active.clear()
        area = 0.0

        for i in range(len(ys) - 1):
            y = ys[i]
            y2 = ys[i + 1]

            for x1, x2, typ in events[y]:
                if typ == 1:
                    active.append((x1, x2))
                else:
                    active.remove((x1, x2))

            width = union_length(active)
            slab_area = width * (y2 - y)

            if area + slab_area >= target:
                # interpolate inside this slab
                return y + (target - area) / width

            area += slab_area

        return ys[-1]  # fallback (should never hit)


# 2943. Maximize Area of Square Hole in Grid (https://leetcode.com/problems/maximize-area-of-square-hole-in-grid/description/) - Medium - 2026-01-15
class Solution:
    '''
    Time Complexity: O(m log m + n log n)
    Space Complexity: O(1)
    Where, m is number of horizontal bars, n is number of vertical bars.
    The key insight is that we can only make a square hole of side S,
    if there are at least S-1 consecutive horizontal bars and S-1 consecutive vertical bars.
    So problem gets reduced to finding maximum consecutive streak in both horizontal and vertical bars.
    Sort the bars and find maximum consecutive difference.
    The side value is minimum of ( maximum consecutive horizontal and vertical gaps ) + 1.
    Square the side value to get maximum area.
    '''
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: List[int], vBars: List[int]) -> int:
        def max_gap(bars):
            bars.sort()
            max_streak = 1
            curr = 1
            
            for i in range(1, len(bars)):
                if bars[i] == bars[i - 1] + 1:
                    curr += 1
                else:
                    curr = 1
                max_streak = max(max_streak, curr)
            
            # gap size = consecutive bars + 1
            return max_streak + 1

        max_h = max_gap(hBars)
        max_v = max_gap(vBars)
        
        side = min(max_h, max_v)
        return side * side


# 2975. Maximum Square Area by Removing Fences From a Field (https://leetcode.com/problems/maximum-square-area-by-removing-fences-from-a-field/description/) - Medium - 2026-01-16
class Solution:
    '''
    Time Complexity: O(h^2 + v^2)
    Space Complexity: O(h^2 + v^2). Actual: O(min(m + n, h^2 + v^2))
    Where, h is number of horizontal fences, v is number of vertical fences (including boundaries 1 and m/n).
    The key insight is that we can only make a square of side S,
    if there exists a horizontal fence at distance S from some other horizontal fence,
    and similarly for vertical fences.
    So we generate all possible horizontal distances and store in a set.
    Then we generate all possible vertical distances and check if it exists in horizontal distances set.
    Return the largest such distance.
    '''
    def maximizeSquareArea(self, m: int, n: int, hFences: List[int], vFences: List[int]) -> int:
        # Include boundary fences
        h = sorted([1] + hFences + [m])
        v = sorted([1] + vFences + [n])

        # All possible horizontal distances
        h_dist = set()
        for i in range(len(h)):
            for j in range(i + 1, len(h)):
                h_dist.add(h[j] - h[i])

        # All possible vertical distances
        v_dist = set()
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                v_dist.add(v[j] - v[i])

        # Find largest common distance
        common = h_dist & v_dist
        if not common:
            return -1

        max_side = max(common)
        return (max_side * max_side) % (10**9 + 7)


# 3047. Find the Largest Area of Square Inside Two Rectangles (https://leetcode.com/problems/find-the-largest-area-of-square-inside-two-rectangles/description/) - Medium - 2026-01-17
class Solution:
    '''
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    Where, n is number of rectangles.
    Brute-force approach to check all pairs of rectangles.
    For each pair, find the overlapping width and height.
    NOTE: Just trying to calculate overlapping width and height directly solves to check if there is any overlap or not, 
    you don't need to check that separately.
    PIT FALLS:
    You first mis-understood the question statement, thinking that if one rectangle had multiple overlaps with other rectangles, 
    then all those overlaps should be considered.
    But actually, only overlaps between any two given rectangles is to be considered. 
    Once you start solving you try to find if there is any intersection between 2 rectangles, 
    as it's process you first wrote down this condition `if (xbj < xti <= xtj and ybj < yti <= ytj) or (xbj <= xbi < xtj and ybj <= ybi < ytj):`.
    But when you started writing code for finding overlapping width and height, 
    you realized that you didn't need that condition at all, as the overlapping width and height calculations 
    would automatically take care of non-overlapping rectangles by resulting in zero width or height.
    So you removed that condition from the final code.
    '''
    def largestSquareArea(self, bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
        area = 0
        n = len(bottomLeft)
        for i in range(n):
            xbi, ybi = bottomLeft[i]
            xti, yti = topRight[i]
            for j in range(i+1, n):
                xbj, ybj = bottomLeft[j]
                xtj, ytj = topRight[j]

                x1 = max(xbi, xbj)
                x2 = min(xti, xtj)
                x = max(x2-x1, 0)
                y1 = max(ybi, ybj)
                y2 = min(yti, ytj)
                y = max(y2-y1, 0)
                s = min(x,y)

                area = max(area, s*s)                    
        return area





############## TEST CASES ##############

# testCases = [
#     ([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"], ["oath","eat"]),
#     ([["a","a"]], ["aa"], ["aa"])
# ]

# for board, words, ans in testCases:
#     print(Solution().findWords(board, words), ans)



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
