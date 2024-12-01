from typing import List
from bisect import bisect_left
from collections import defaultdict

class Solution:
    def minArraySum2(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        '''
        Couldn't figure out the corner case on my own, below solution doesn't work!
        Solution using greedy approach that works: https://leetcode.com/problems/minimum-array-sum/solutions/6078002/o-n-log-n-greedy
        '''
        nums.sort()
        largeNums = bisect_left(nums, 2*k-1)
        mediumNums = bisect_left(nums, k)

        i = len(nums)-1
        while op1 and i >= largeNums:
            nums[i] = (nums[i]+1)//2
            op1 -= 1
            if op2:
                nums[i] -= k
                op2 -= 1
            i -= 1

        j = mediumNums
        while op2 and j <= i:
            nums[j] -= k
            op2 -= 1
            j += 1

        nums = sorted(nums[:i+1]) + nums[i+1:]
        while op1 and i > -1:
            if nums[i] < 2: break
            nums[i] = (nums[i]+1) //2
            op1 -= 1
            i -= 1

        return sum(nums)
    
    
    # Solution 2 using dynamic programming
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        '''
        Works flawlessly.
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





if __name__ == "__main__":
    sol = Solution()
    questions = [
        (
            ([1, 3, 5, 7, 9, 12, 12, 12, 13, 15, 15, 15, 16, 17, 19, 20], 11, 15, 4),
            77
        ),
        (
            ([5,5], 1, 1, 2),
            6
        ),
        (
            ([1], 1, 0, 1),
            0
        ),
        (
            ([0,7,0,2,3], 2, 4, 4), 
            2
        ),
        (
            ([2,10,9,0,4], 3, 5, 2), 
            7
        ),
        (
            ([7,4,4,8], 3, 3, 1),
            11
        ),
        (
            ([5], 2, 1, 0), 
            3
        )
    ]

    for question, answer in questions:
        original = list(question[0])
        result = sol.minArraySum(*question)
        if result == answer:
            print(f"Success! {original} -> {result}")
        else:
            print(f"Failed! {original} -> {result} (answer: {answer})")