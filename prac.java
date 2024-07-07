import java.util.*;

// 125
class Solution {
    public boolean isAlNum(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); 
    }

    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        int left = 0, right = s.length()-1;
        while (left < right) {
            while ( !isAlNum(s.charAt(left)) && left < right ) {
                left++;
            }
            while ( !isAlNum(s.charAt(right)) && right > left ) {
                right--;
            }
            if ( s.charAt(left) != s.charAt(right) ) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}


class Main {
    public static void main(String[] args) {
         ArrayList<List<String>> questions = new ArrayList<List<String>>();
        // questions.add(Arrays.asList("eat", "tea", "tan", "ate", "nat", "bat"));
        // questions.add(Arrays.asList("e", "at", "ta", "ate", ""));                
    
        // Solution sol = new Solution();
        // for (List<String> question: questions) {
        //     System.out.println(sol.groupAnagrams(question.toArray(new String[0])));
        // }
    };        
}


// class Solution {
//     public static int maxProductDifference(int[] nums) {
//         int max1 = 0, max2 = 0, min1 = 100000, min2 = 100000;
//         for(int num: nums){
//             if (num > max1) {max2=max1; max1=num;} else if (num > max2) {max2=num;}
//             if (num < min1) {min2=min1; min1=num;} else if (num < min2) {min2=num;}
//         }
//         return max2*max1 - min1*min2;
//     }

//     public static  int longestConsecutive(int[] nums) {
//         if (nums.length == 0) return 0;
//         Arrays.sort(nums);
//         int prev = nums[0];
//         int seq = 1;
//         int maxSeq = 1;
//         for (int num : nums) {
//             if (prev + 1 == num) {
//                 seq++;
//                 maxSeq = Math.max(maxSeq, seq);
//             }  else if (prev != num) {
//                 seq = 1;
//             }
//             prev = num;
//         }
//         return maxSeq;
//     }

//     public static void main(String[] args) {  
//         int[][] questions = {{1, 2, 4,4 ,4,7,8}, {2, 4, 5, 9, 1, 10}, {1,2,0,1, 5, 7, 3}};
//         for (int[] question : questions) {
//             System.out.println(maxProductDifference(question));
//         }
//     }
// }
