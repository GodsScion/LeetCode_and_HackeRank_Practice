from types import List, Optional
from collections import defaultdict, Counter
import re
import bisect

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




#######  SLIDING WINDOW  #######
# 76. Minimum Window Substring (https://leetcode.com/problems/minimum-window-substring/description/) - Hard
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




###### BACKTRACKING ######
# 39. Combination Sum (https://leetcode.com/problems/combination-sum/description/) - Medium
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        self.candidates = candidates
        self.target = target
        return self.getCombination(0, 0, [])

    def getCombination(self, limit, cur, combination):
        if cur == self.target:
            return [combination]
        output = []
        while limit < len(self.candidates) and self.candidates[limit] + cur <= self.target:
            output += self.getCombination(limit, cur + self.candidates[limit], combination + [self.candidates[limit]])
            limit += 1
        return output

# 78. Subsets (https://leetcode.com/problems/subsets/description/) - Medium
# NOT THE  MOST EFFICIENT SOLUTION
from copy import deepcopy
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        output = [[]]
        for n in nums:
            l = len(output)
            output = output + deepcopy(output)
            for i in range(l):
                output[i].append(n)
        return output



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
                



#######  DYNAMIC PROGRAMMING  #######

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

####################################







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
