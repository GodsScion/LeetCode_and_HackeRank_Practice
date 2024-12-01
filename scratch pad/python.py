from typing import List
from bisect import bisect_left

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        nums.sort()
        largeNums = max(bisect_left(nums, 2*k-1), len(nums) - min(op1, op2))
        mediumNums = bisect_left(nums, k)

        i = len(nums)-1
        while op1 and i >= largeNums:
            nums[i] = (nums[i]+1)//2
            op1 -= 1
            if op2:
                nums[i] -= k
                op2 -= 1
            i -= 1

        i = mediumNums
        while op2 and i < largeNums:
            nums[i] -= k
            op2 -= 1
            i += 1

        nums = sorted(nums[:largeNums]) + nums[largeNums:]
        i = largeNums-1
        while op1 and i > -1:
            if nums[i] < 2: break
            nums[i] = (nums[i]+1) //2
            op1 -= 1
            i -= 1

        return sum(nums)
    
## Chatgpt sol doesn't work
    # def minArraySum2(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    #     # Store the potential impact of operations on each number
    #     reductions = []

    #     for i, num in enumerate(nums):
    #         # Calculate results of applying the operations
    #         # Only Operation 1
    #         op1_only = (num + 1) // 2
    #         # Only Operation 2
    #         op2_only = num - k if num >= k else num
    #         # Both operations in different orders
    #         both_op1_then_op2 = (num + 1)//2 - k if (num + 1)//2 >= k else (num + 1)//2
    #         both_op2_then_op1 = (num - k + 1) // 2 if num >= k else num

    #         # Choose the best result for this number
    #         best_result = min(op1_only, op2_only, both_op1_then_op2, both_op2_then_op1)
    #         reductions.append((num - best_result, i, best_result))  # (reduction, index, resulting value)

    #     # Sort by reduction in descending order
    #     reductions.sort(reverse=True, key=lambda x: x[0])

    #     # Track which operations have been used
    #     used_op1 = set()
    #     used_op2 = set()
    #     result = nums[:]

    #     for reduction, index, new_value in reductions:
    #         if op1 > 0 and index not in used_op1:
    #             if result[index] > new_value:
    #                 result[index] = new_value
    #                 op1 -= 1
    #                 used_op1.add(index)
    #         elif op2 > 0 and index not in used_op2:
    #             if result[index] > new_value:
    #                 result[index] = new_value
    #                 op2 -= 1
    #                 used_op2.add(index)

    #     return sum(result)



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
        result = sol.minArraySum2(*question)
        if result == answer:
            print(f"Success! {original} -> {result}")
        else:
            print(f"Failed! {original} -> {result} (answer: {answer})")