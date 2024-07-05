import java.util.*;

// 49
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,ArrayList<String>> hash = new HashMap<>();
        for(String word: strs) {
            char[] count = new char[26];
            for(char c: word.toCharArray()) {
                count[c-'a']++;
            }
            String id = String.valueOf(count);
            hash.putIfAbsent(id, new ArrayList<String>());
            hash.get(id).add(word);
        }
        return new ArrayList<>(hash.values());
    }
}

class Main {
    public static void main(String[] args) {
        ArrayList<List<String>> questions = new ArrayList<List<String>>();
        questions.add(Arrays.asList("eat", "tea", "tan", "ate", "nat", "bat"));
        questions.add(Arrays.asList("e", "at", "ta", "ate", ""));                
    
        Solution sol = new Solution();
        for (List<String> question: questions) {
            System.out.println(sol.groupAnagrams(question.toArray(new String[0])));
        }
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
