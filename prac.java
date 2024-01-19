import java.util.Arrays;

class Solution {
    public static  int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Arrays.sort(nums);
        int prev = nums[0];
        int seq = 1;
        int maxSeq = 1;
        for (int num : nums) {
            if (prev + 1 == num) {
                seq++;
                maxSeq = Math.max(maxSeq, seq);
            } else {
                seq = 1;
            }
            prev = num;
        }
        return maxSeq;
    }

    public static void main(String[] args) {  
        int[][] questions = {{1, 2, 4}, {2, 4, 5}};
        for (int[] question : questions) {
            System.out.println(longestConsecutive(question));
        }
    }
}
