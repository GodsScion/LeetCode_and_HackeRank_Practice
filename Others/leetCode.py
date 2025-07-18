from traceback import print_tb
from typing import Optional
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if self.isSametree(root, subRoot):
            return True
        if root != None:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        return False

    def isSametree(self,  root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 == None and root2 == None:
            return True
        if root1 == None or root2 == None:
            return False
        return self.isSametree(root1.left, root2.left) and self.isSametree(root1.right, root2.right)

root = TreeNode(1)
subRoot = TreeNode(0)
print(Solution().isSubtree(root, subRoot))

# from bisect import bisect_left, bisect_right

# def prob(nums):
#         nums.sort()
#         print(f"Sorted array: \n{nums}\n")
#         mid = bisect_left(nums, 0)
#         end = bisect_right(nums, -1*min(nums[0]+nums[1],nums[0]))-1
#         print(f"Mid: {nums[mid]} -> {mid};")
        
#         output = []
#         while end >= mid:
#             start = bisect_left(nums, -nums[end] -nums[end-1])
#             print(f">>> Current values are, Start: {nums[start]} -> {start}; End: {nums[end]} -> {end}")
#             while start < mid:
#                 target = -nums[end] - nums[start]
#                 print(f"Target: {target} = -({nums[start]} + {nums[end]}); Start -> {start}")
#                 if target < nums[start]:
#                     print(f"Changing end value, {target} > Start {nums[start]}")
#                     break
#                 if nums[bisect_right(nums, target, start+2, end)-1] == target:
#                     output.append( [nums[start], target, nums[end]] )
#                     print(f"* Added.. {[nums[start], target, nums[end]]}")
#                 while start < mid and nums[start] == nums[start+1]: start+=1
#                 start += 1
#             while end > mid and nums[end] == nums[end-1]: end-=1
#             end -= 1
#         if bisect_right(nums, 0) - mid > 2: output.append([0,0,0])
#         return output

# inputs = [
#             # ([1,2,-2,-1], []), 
#             # ([-1,0,1,2,-1,-4], [[-1,-1,2],[-1,0,1]]),
#             ([6,-5,-6,-1,-2,8,-1,4,-10,-8,-10,-2,-4,-1,-8,-2,8,9,-5,-2,-8,-9,-3,-5], [[-10,4,6],[-8,-1,9],[-6,-3,9],[-6,-2,8],[-5,-4,9],[-5,-3,8],[-5,-1,6],[-4,-2,6],[-3,-1,4],[-2,-2,4]])
#         ]

# for l in inputs:
#         output = prob(l[0])
#         print(f"\nFailed!\nOutput: {sorted(output)}\nExpected: {sorted(l[1])}") if sorted(output) != sorted(l[1]) else print(f"\nSuccess. {output}")