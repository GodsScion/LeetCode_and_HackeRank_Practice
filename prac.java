import java.util.*;

class Solution {
    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Arrays.sort(nums);
        int prev = nums[0];
        int seq = 1;
        int maxSeq = 1;
        for (int num : nums) {
            if (prev + 1 == num) {
                seq++;
                maxSeq = Math.max(maxSeq, seq);
            } else if (prev != num) {
                seq = 1;
            }
            prev = num;
        }
        return maxSeq;
    }
}