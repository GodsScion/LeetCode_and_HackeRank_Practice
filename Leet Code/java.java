import java.util.*;

class Solution {
    //#######  ARRAYS AND HASHING  #######//
    // 217. Contains Duplicate (https://leetcode.com/problems/contains-duplicate/description/) - Easy
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> prevNums = new HashSet<Integer>();
        for(int num: nums) {
            if(prevNums.contains(num)) {
                return true;
            }
            prevNums.add(num);
        }
        return false;
    }

    // 242. Valid Anagram (https://leetcode.com/problems/valid-anagram/description/) - Easy
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) { return false; }
        int[] count = new int[26];
        for (int i=0; i<s.length(); i++) {
            count[s.charAt(i) - 'a']++;
            count[t.charAt(i) - 'a']--;
        }
        for (int i=0; i<26; i++) {
            if(count[i] != 0) {
                return false;
            }
        }
        return true;
    }

    // 1. Two Sum (https://leetcode.com/problems/two-sum/description/) - Easy
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> prev = new HashMap<>();
        for (int i=0; i<nums.length; i++) {
            int need = target-nums[i];
            if (prev.containsKey(need)) {
                return new int[] { prev.get(need) , i};
            }
            prev.put(nums[i], i);
        }
        return null;
    }

    // 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String,List<String>> hash_map = new HashMap<>();
        for (String word: strs) {
            char[] id = word.toCharArray();
            Arrays.sort(id);
            String wordId = new String(id);

            if (!hash_map.containsKey(wordId)) {
                hash_map.put(wordId, new ArrayList<String>());
            }
            hash_map.get(wordId).add(word);
        }

        return new ArrayList<>(hash_map.values());
    }
    // 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
    public List<List<String>> groupAnagrams2(String[] strs) {
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
    // 49. Group Anagrams (https://leetcode.com/problems/group-anagrams/description/) - Medium
    public List<List<String>> groupAnagrams3(String[] strs) {
        Map<String, ArrayList<String>> map = new HashMap<>();
        for (String word: strs) {
            int[] count = new int[26];
            for (char c: word.toCharArray()) {
                count[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i: count) {
                sb.append(i).append("#");
            }
            String id = sb.toString();
            if(!map.containsKey(id)) {
                map.put(id, new ArrayList<String>());
            }
            map.get(id).add(word);
        }
        return new ArrayList<List<String>>(map.values());
    }

    // 347. Top K Frequent Elements (https://leetcode.com/problems/top-k-frequent-elements/description/) - Medium
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }

        PriorityQueue<Map.Entry<Integer, Integer>> maxHeap = new PriorityQueue<>(
            (a, b) -> b.getValue() - a.getValue()
        );

        maxHeap.addAll(freq.entrySet());
        
        int[] output = new int[k];
        for (int i = 0; i < k; i++) {
            output[i] = maxHeap.poll().getKey();
        }

        return output;
    }
    // 347. Top K Frequent Elements (https://leetcode.com/problems/top-k-frequent-elements/description/) - Medium
    public int[] topKFrequent2(int[] nums, int k) {
        
        List<List<Integer>> buckets = new ArrayList<>();
        for (int i=0; i<=nums.length; i++) {
            buckets.add(new ArrayList<>());
        }

        Map<Integer, Integer> freq = new HashMap<>();
        for (int num: nums) {
            freq.put(num, freq.getOrDefault(num,0) + 1);
        }

        for (int num: freq.keySet()) {
            int count = freq.get(num);
            buckets.get(count).add(num);
        }

        int[] output = new int[k];
        for (int i = buckets.size()-1; i >= 0 ; i--) {
            if(buckets.get(i) != null) {
                for(int num: buckets.get(i)) {
                    if (k==0) {
                        return output;
                    }
                    output[k-1] = num;
                    k--;
                }
            } 
        }

        return output;
    }

    // 238. Product of Array Except Self (https://leetcode.com/problems/product-of-array-except-self/description/) - Medium
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] back = new int[n];
        back[n-1] = nums[n-1];
        for (int i = n-2; i > 0; i--) {
            back[i] = nums[i] * back[i+1];
        }
        for (int i = 1; i < n; i++) {
            nums[i] = nums[i] * nums[i-1];
        }
        int[] output = new int[n];
        output[0] = back[1];
        output[n-1] = nums[n-2];
        for (int i = 1; i < n-1; i++) {
            output[i] = nums[i-1] * back[i+1];
        }
        return output;
    }
    // 238. Product of Array Except Self (https://leetcode.com/problems/product-of-array-except-self/description/) - Medium
    public int[] productExceptSelf2(int[] nums) {
        int n = nums.length;
        int[] arr = new int[n];
        int left = 1, right = 1;
        for (int i=0; i<n; i++) {
            arr[i] = left;
            left *= nums[i]; 
        }
        for (int i = n - 1; i >= 0; i--) {
            arr[i] *= right;
            right *= nums[i];
        }
        return arr;
    }

    // 36. Valid Sudoku (https://leetcode.com/problems/valid-sudoku/description/) - Medium
    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            int[] rowDuplicates = new int[10];
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    if (rowDuplicates[board[i][j] - '0'] > 0) {
                        return false;
                    }
                    rowDuplicates[board[i][j] - '0']++;
                }
            }
        }
        for (int i = 0; i < 9; i++) {
            int[] colDuplicates = new int[10];
            for (int j = 0; j < 9; j++) {
                if (board[j][i] != '.') {
                    if (colDuplicates[board[j][i] - '0'] > 0) {
                        return false;
                    }
                    colDuplicates[board[j][i] - '0']++;
                }
            }
        }
        for (int i = 0; i < 9; i+=3) {
            for (int j = 0; j < 9; j+=3) {
                int[] boxDuplicates = new int[10];
                for (int r = i; r < i + 3; r++) {
                    for (int c = j; c < j + 3; c++) {
                        if (board[r][c] != '.') {
                            if (boxDuplicates[board[r][c] - '0'] > 0) {
                                return false;
                            }
                            boxDuplicates[board[r][c] - '0']++;
                        }
                    }
                }
            }
        }
        return true;
    }
    // 36. Valid Sudoku (https://leetcode.com/problems/valid-sudoku/description/) - Medium
    public boolean isValidSudoku2(char[][] board) {
        for (int i=0; i<9; i++) {
            Set<Character> row = new HashSet<>();
            Set<Character> col = new HashSet<>();
            Set<Character> box = new HashSet<>();
            for (int j=0; j<9; j++) {
                if (board[i][j] != '.' && !row.add(board[i][j])) {
                    return false;
                }
                if (board[j][i] != '.' && !col.add(board[j][i])) {
                    return false;
                }
                if (board[3*(i/3) + j/3][3*(i%3) + j%3] != '.' && !box.add(board[3*(i/3) + j/3][3*(i%3) + j%3])) {
                    return false;
                }
            }
        }
        return true;
    }
    // 36. Valid Sudoku (https://leetcode.com/problems/valid-sudoku/description/) - Medium
    public boolean isValidSudoku3(char[][] board) {
        for (int i=0; i<9; i++) {
            int[] row = new int[9];
            int[] col = new int[9];
            int[] box = new int[9];
            for (int j=0; j<9; j++) {
                if (board[i][j] != '.') {
                    if(row[board[i][j] - '1'] > 0) {
                        return false;
                    }
                    row[board[i][j] - '1']++;
                }
                if (board[j][i] != '.') {
                    if(col[board[j][i] - '1'] > 0) {
                        return false;
                    }
                    col[board[j][i] - '1']++;
                }
                int r = 3*(i/3) + j/3;
                int c = 3*(i%3) + j%3;
                if (board[r][c] != '.') {
                    if (box[board[r][c] - '1'] > 0) {
                        return false;
                    }
                    box[board[r][c] - '1']++;
                }
            }
        }
        return true;
    }
    


    // 128. Longest Consecutive Sequence (https://leetcode.com/problems/longest-consecutive-sequence/description/) - Medium
    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) { return 0; }
        Arrays.sort(nums);
        int max = 1;
        int count = 1;
        for (int i=1; i<nums.length; i++) {
            if (nums[i] == nums[i-1] + 1) {
                count++;
            } else if (nums[i] == nums[i-1]) {
                continue;
            } else {
                max = Math.max(count, max);
                count = 1;
            }
        }
        return Math.max(count,max);
    }
    // 128. Longest Consecutive Sequence (https://leetcode.com/problems/longest-consecutive-sequence/description/) - Medium
    public int longestConsecutive2(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        
        int maxCount = 0;
        for (int num: nums) {
            if (!set.contains(num-1)) {
                int count = 1;
                while (set.contains(++num)) {
                    count++;
                }
                maxCount = Math.max(count, maxCount);
            }
        }
        return maxCount;
    }

    //#######  TWO POINTERS  #######//
    // 125. Valid Palindrome (https://leetcode.com/problems/valid-palindrome/description/) - Easy
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

    // 167. Two Sum II - Input Array Is Sorted (https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) - Medium
    public int[] twoSumII(int[] numbers, int target) {
        /* 
         * Change function name from twoSumII to twoSum
        */
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int current = numbers[left] + numbers[right];
            if (current == target) {
                return new int[] { left + 1, right + 1 };
            }
            if (current > target) {
                right--;
            } else {
                left++;
            }
        }
        return new int[] {};
    }
    
    // 15. 3Sum (https://leetcode.com/problems/3sum/description/) - Medium
    public List<List<Integer>> threeSum(int[] nums) {
        int len = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> out = new ArrayList<>();

        for (int i=0; i<len; i++) {
            if (i > 0 && nums[i-1] == nums[i]) { continue; }
            int left = i+1;
            int right = len-1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if ( sum == 0 ) {
                    out.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    left++;
                    while (left <= right && nums[left-1] == nums[left]) {
                        left++;
                    }
                } else if (sum > 0) {
                    right--;
                } else {
                    left++;
                }
            }
        }
        return out;
    }

    // 11. Container With Most Water (https://leetcode.com/problems/container-with-most-water/description/) - Medium
    public int maxArea(int[] height) {
        int max = 0, left = 0, right = height.length-1;
        while (left < right) {
            if ( height[left] < height[right] ) {
                max = Math.max(max, height[left]*(right-left));
                left++;
            } else {
                max = Math.max(max, height[right]*(right-left));
                right--;
            }
        }
        return max;
    }

    //#######  SLIDING WINDOW  #######//
    // 121. Best Time to Buy and Sell Stock (https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/) - Easy
    public int maxProfit(int[] prices) {
        int max = 0, buy = prices[0];
        for (int i=1; i<prices.length; i++) {
            if (prices[i] < buy) {
                buy = prices[i];
            } else {
                max = Math.max(prices[i] - buy, max);
            }
        }
        return max;
    }
    public int maxProfit2(int[] prices) {
        int max = 0, buy = prices[0];
        for (int price: prices) {
            buy = Math.min(buy, price);
            max = Math.max(price - buy, max);
        }
        return max;
    }

    // 3. Longest Substring Without Repeating Characters (https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) - Medium
    // // Best Solution
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, right = 0, max = 0;
        while (right < s.length()) {
            if (!set.contains(s.charAt(right))) {
                set.add(s.charAt(right));
                max = Math.max(max, set.size());
                right++;
            } else {
                set.remove(s.charAt(left));
                left++;
            }
        }
        return max;
    }
    // 3. Longest Substring Without Repeating Characters (https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) - Medium
    // // Easy to understand
    public int lengthOfLongestSubstring2(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, right = 0, max = 0;
        while ( right < s.length() ) {
            while (set.contains(s.charAt(right))) {
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            max = Math.max(max, set.size());
            right++;
        }
        return max;
    }
    // 3. Longest Substring Without Repeating Characters (https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) - Medium
    // // Easy to understand - Set is optimized
    public int lengthOfLongestSubstring3(String s) {
        boolean[] visited = new boolean[256];
        int left = 0, max = 0;
        for (int right = 0; right < s.length(); right++) {
            while (visited[s.charAt(right)]) {
                visited[s.charAt(left)] = false;
                left++;
            }
            visited[s.charAt(right)] = true;
            max = Math.max(max, right-left+1);
        }
        return max;
    }

    // 424. Longest Repeating Character Replacement (https://leetcode.com/problems/longest-repeating-character-replacement/description/) - Medium
    public int characterReplacement(String s, int k) {
        int[] freq = new int[26];
        int start = 0, count = 0;
        for (int end = 0; end < s.length(); end++) {
            freq[s.charAt(end) - 'A']++;
            count = Math.max(count, freq[s.charAt(end) - 'A']);
            if ( end - start + 1  >  count + k ) {
                freq[s.charAt(start) - 'A']--;
                start++;
            }
        }
        return s.length() - start;
    }

    // 76. Minimum Window Substring (https://leetcode.com/problems/minimum-window-substring/description/) - Hard
    public String minWindow(String s, String t) {
        if (t.length() > s.length()) { return ""; }
        int need = 0, have = 0, i = 0, oi = 0, oj = s.length()+t.length();
        Map<Character, Integer> want = new HashMap<>();
        Map<Character, Integer> window = new HashMap<>();
        List<Integer> indices = new ArrayList<>();

        for (char c: t.toCharArray()) {
            want.put(c, want.getOrDefault(c,0) + 1);
        }
        need = want.size();

        for (int k=0; k<s.length(); k++) {
            char c = s.charAt(k);
            if (want.containsKey(c)) {
                indices.add(k);
                window.put(c, window.getOrDefault(c, 0) + 1 );
                if (window.get(c).equals(want.get(c))) {
                    have++;
                }
                if (need == have) {
                    while (indices.get(i) < k && window.get(s.charAt(indices.get(i))) > want.get(s.charAt(indices.get(i))) ) {
                        c = s.charAt(indices.get(i));
                        window.put(c ,window.get(c)-1);
                        i++;
                    }
                    if (oj-oi > k-indices.get(i)) {
                        oj = k;
                        oi = indices.get(i);
                    }
                    c = s.charAt(indices.get(i));
                    window.put(c,window.get(c)-1);
                    have--;
                    i++;
                }
            }
        }

        if (oj == s.length()+t.length()) {
            return "";
        }
        return s.substring(oi,oj+1);
    }
    // 76. Minimum Window Substring (https://leetcode.com/problems/minimum-window-substring/description/) - Hard
    // // Similar logic but is quicker because of using arrays as hashmaps
    public String minWindow2(String s, String t) {
        if (t.length() > s.length()) {return "";}
        int[] need = new int[256];
        int[] window = new int[256];
        
        for (char c : t.toCharArray()) {
            need[c]++;
        }
        
        int left = 0;
        int right = 0;
        int minLen = s.length() + 1;
        int minLeft = 0;
        
        int count = 0;
        
        while (right < s.length()) {
            char rightChar = s.charAt(right);
            window[rightChar]++;

            if (window[rightChar] <= need[rightChar]) {
                count++;
            }

            while (count == t.length()) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minLeft = left;
                }

                char leftChar = s.charAt(left);
                window[leftChar]--;
                
                if (window[leftChar] < need[leftChar]) {
                    count--;
                }
                
                left++;
            }
            
            right++;
        }
        
        if (minLen == s.length() + 1) {
            return "";
        }

        return s.substring(minLeft, minLeft + minLen);
    }
        
    
    //#######  STACK  #######//
    // 20. Valid Parentheses (https://leetcode.com/problems/valid-parentheses/description/) - Easy
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        String par = "{([";
        for (char c: s.toCharArray()) {
            if (par.indexOf(c) != -1) {
                stack.push(c);
            } else {
                switch(c) {                
                    case '}':
                        if (stack.isEmpty() || stack.pop() != '{') return false;
                        break;
                    case ']':
                        if (stack.isEmpty() || stack.pop() != '[') return false;
                        break;
                    case ')':
                        if (stack.isEmpty() || stack.pop() != '(') return false;
                        break;
                }
            }
        }
        return stack.isEmpty();
    }

    // 155. Min Stack (https://leetcode.com/problems/min-stack/description/) - Medium
    // // Competitive level optimized solution, kind of overkill for real world scenarios
    /**
     * Your MinStack object will be instantiated and called as such:
     * MinStack obj = new MinStack();
     * obj.push(val);
     * obj.pop();
     * int param_3 = obj.top();
     * int param_4 = obj.getMin();
     */
    class MinStack {
        ListNode list;
        int min = Integer.MAX_VALUE;
        public MinStack() {
            list = null;
        }
        
        public void push(int val) {
            min = Math.min(val, min);
            ListNode node = new ListNode(val, list);
            list = node;
        }
        
        public void pop() {
            int val = list.val;
            list = list.next;
            if(val == min) {
                min = Integer.MAX_VALUE;
                ListNode dummy = list;
                while(dummy != null) {
                    min = Math.min(min, dummy.val);
                    dummy = dummy.next;
                }
            }
        }
        
        public int top() {
            return list.val;
        }
        
        public int getMin() {
            return min;
        }
    }

    // 155. Min Stack (https://leetcode.com/problems/min-stack/description/) - Medium
    // Typical Solution
    // // This solution can be considered as optimized for interview purposes, but for competitive coding level use ListNode solution. 
    /**
     * Your MinStack object will be instantiated and called as such:
     * MinStack obj = new MinStack();
     * obj.push(val);
     * obj.pop();
     * int param_3 = obj.top();
     * int param_4 = obj.getMin();
     */
    class MinStack2 {
        private Stack<Integer> stack;
        private Stack<Integer> minStack;

        public MinStack2() {
            stack = new Stack<>();
            minStack = new Stack<>();
        }

        public void push(int val) {
            stack.push(val);
            if (minStack.isEmpty()) {
                minStack.push(val);
            } else {
                minStack.push(Math.min(val, minStack.peek()));
            }
        }

        public void pop() {
            stack.pop();
            minStack.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }




    //#######  BINARY SEARCH  #######//
    // 153. Find Minimum in Rotated Sorted Array (https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/) - Medium
    public int findMin(int[] nums) {
        int left = 0, right = nums.length-1, mid;
        while (left < right) {
            mid = (right + left)/2;
            if (nums[left] > nums[mid]) {
                right = mid;
                left++;
            } 
            else if (nums[mid] > nums[right]) {
                left = mid+1;
            }
            else {
                return nums[left];
            }
        }
        return nums[right];
    }
    
    // 33. Search in Rotated Sorted Array (https://leetcode.com/problems/search-in-rotated-sorted-array/description/) - Medium
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length-1, mid;
        while (left <= right) {
            mid = (left + right)/2;
            if (target == nums[mid]) { return mid; }

            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid-1;
                } else {
                    left = mid+1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid+1;
                } else {
                    right = mid-1;
                }
            }
        }
        return -1;
    }

    
    //#######  LINKED LIST  #######//
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    // 206. Reverse Linked List (https://leetcode.com/problems/reverse-linked-list/description/) - Easy
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    // 21. Merge Two Sorted Lists (https://leetcode.com/problems/merge-two-sorted-lists/description/) - Easy
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode cur = dummy;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if (list1 != null) {
            cur.next = list1;
        } else {
            cur.next = list2;
        }
        return dummy.next;
    }

    // 143. Reorder List (https://leetcode.com/problems/reorder-list/description/) - Medium
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public void reorderList(ListNode head) {
        ListNode slowPointer = head;
        ListNode fastPointer = head;
        while (fastPointer.next != null) {
            fastPointer = fastPointer.next;
            if (fastPointer.next != null) {
                fastPointer = fastPointer.next;
                slowPointer = slowPointer.next;
            }
        }
        
        ListNode prev = null;
        ListNode cur = slowPointer.next;
        slowPointer.next = null;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }

        slowPointer = head;
        while (slowPointer != null && fastPointer != null) {
            cur = slowPointer.next;
            slowPointer.next = fastPointer;
            slowPointer = cur;
            prev = fastPointer.next;
            fastPointer.next = cur;
            fastPointer = prev;
        }
    }

    // 19. Remove Nth Node From End of List (https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) - Medium
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode end = dummy;
        ListNode nNode = dummy;
        for (int i=0; i<=n; i++) {
            end = end.next;
        }
        while (end != null) {
            nNode = nNode.next;
            end = end.next;
        }
        nNode.next = nNode.next.next;
        return dummy.next;
    }

    // 141. Linked List Cycle (https://leetcode.com/problems/linked-list-cycle/description/) - Easy
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public boolean hasCycle(ListNode head) {
        ListNode runner = head, chaser = head;
        while (runner != null && runner.next != null) {
            runner = runner.next.next;
            chaser = chaser.next;
            if (runner == chaser) {
                return true;
            }
        }
        return false;
    }

    // 23. Merge k Sorted Lists (https://leetcode.com/problems/merge-k-sorted-lists/description/) - Hard
    // // Priority Queue (As quick as Divide and Conquer, theoretical time complexity is best)
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = new ListNode();
        ListNode current = dummy;
        Queue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
                @Override
                public int compare(ListNode l1, ListNode l2) {
                    return l1.val - l2.val;
                }
            });
        
        for (ListNode list: lists) {
            if (list != null) {
                pq.add(list);
            }
        }

        while (pq.size() > 1) {
            current.next = pq.poll();
            current = current.next;
            if (current.next != null) {
                pq.add(current.next);
            }
        }

        current.next = pq.poll();

        return dummy.next;
    }

    // 23. Merge k Sorted Lists (https://leetcode.com/problems/merge-k-sorted-lists/description/) - Hard
    // // Divide and Conquer (Better than Priority Queue in real world, not by theoretical time complexity)
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode() {}
     *     ListNode(int val) { this.val = val; }
     *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return mergeKLists(lists, 0, lists.length - 1);
    }
    private ListNode mergeKLists(ListNode[] lists, int start, int end) {
        if (start == end) return lists[start];
        int mid = start + (end - start) / 2;
        ListNode left = mergeKLists(lists, start, mid);
        ListNode right = mergeKLists(lists, mid + 1, end);
        return mergeTwoListsHelper(left, right);
    }
    private ListNode mergeTwoListsHelper(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode();
        ListNode current = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }
        if (l1 != null) current.next = l1;
        if (l2 != null) current.next = l2;
        return dummy.next;
    }



    //#######  TREES  #######//
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // 226. Invert Binary Tree (https://leetcode.com/problems/invert-binary-tree/description/) - Easy
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode temp = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(temp);
        return root;
    }

    // 104. Maximum Depth of Binary Tree (https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) - Easy
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // 100. Same Tree (https://leetcode.com/problems/same-tree/description/) - Easy
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null || p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    // 572. Subtree of Another Tree (https://leetcode.com/problems/subtree-of-another-tree/description/) - Easy
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        /*
         * Time complexity: O(n*m)
         * Space complexity: O(n) for the recursion stack.
         * Where, `n` is the number of nodes in `root` and `m` is the number of nodes in `subRoot`.
         */
        if (isSametree(root, subRoot)) {
            return true;
        }
        if (root == null) {
            return false;
        }
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }
    public boolean isSametree(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null || root1.val != root2.val) {
            return false;
        }
        return isSametree(root1.left, root2.left) && isSametree(root1.right, root2.right);
    }

    // 235. Lowest Common Ancestor of a Binary Search Tree (https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) - Medium
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode(int x) { val = x; }
     * }
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        }
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return root;
    }
    
    // 98. Validate Binary Search Tree (https://leetcode.com/problems/validate-binary-search-tree/description/) - Medium
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    public boolean isValidBST(TreeNode root) {
        return isValidChild(root, null, null);
    }
    public boolean isValidChild(TreeNode node, Integer minLimit, Integer maxLimit) {
        /*
        * Do not fall into a pitfall by hardcoding Min and Max limits with Integer.MAX_VALUE.
        * If you do that, and if one of the test cases have their root value as Integer.MAX_VALUE, it would return false, although the BST is valid.
        * Example test case: [Integer.MAX_VALUE]
        * A single node BST with max int value is valid!
        */
        if (node == null) {
            return true;
        }
        if ((minLimit != null && node.val <= minLimit) || (maxLimit != null && node.val >= maxLimit)) {
            return false;
        }
        return isValidChild(node.left, minLimit, node.val) && isValidChild(node.right, node.val, maxLimit);
    }

    // 230. Kth Smallest Element in a BST (https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) - Medium
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    public int kthSmallest(TreeNode root, int k) {
        List<TreeNode> stack = new LinkedList<>();
        while (k > 0) {
            while (root.left != null) {
                stack.addLast(root);
                root = root.left;
            }
            k -= 1;
            if (k == 0) {
                return root.val;
            }
            if (root.right != null) {
                root = root.right;
            } else {
                root = stack.removeLast();
                root.left = null;
            }
        }
        return root.val;
    }

    // 105. Construct Binary Tree from Preorder and Inorder Traversal (https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - Medium
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    private int[] preorder;
    private int[] inorder;
    private HashMap<Integer, Integer> index;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        this.inorder = inorder;
        this.index = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            this.index.put(inorder[i], i);
        }
        return tree(0, preorder.length, 0, inorder.length);
    }
    public TreeNode tree(int p1, int p2, int i1, int i2) {
        if ( p1 == p2 ) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[p1]);
        int len = index.get(root.val) - i1 + 1;
        root.left = tree(p1 + 1, p1 + len, i1, i1 + len);
        root.right = tree(p1 + len, p2, i1 + len, i2);
        return root;
    }

    // 124. Binary Tree Maximum Path Sum (https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) - Hard
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */
    int max = -10000;
    public int maxPathSum(TreeNode root) {
        maxWithChild(root);
        return max;
    }
    public int maxWithChild(TreeNode root) {
        if (root == null) return -10000;
        int left = maxWithChild(root.left);
        int right = maxWithChild(root.right);
        int nodeMax = Math.max(root.val, root.val + Math.max(left, right));
        max = Math.max(max, Math.max(nodeMax, root.val + left + right));
        return nodeMax;
    }


    
    //#######  2-D DYNAMIC PROGRAMMING  #######//
    // 62. Unique Paths (https://leetcode.com/problems/unique-paths/description/) - Medium
    int[][] visited;
    public int uniquePaths(int m, int n) {
        visited = new int[m][n];
        for(int r=0; r<m; r++) {
            visited[r][n-1] = 1;
        }
        for(int c=0; c<n; c++) {
            visited[m-1][c] = 1;
        }
        return getUniquePaths(0, 0, m, n);
    }
    private int getUniquePaths(int r, int c, int m, int n) {
        if (r >= m | c >= n) {
            return 0;
        }
        if (visited[r][c] > 0) {
            return visited[r][c];
        }
        visited[r][c] = getUniquePaths(r+1, c, m, n) + getUniquePaths(r, c+1, m, n);
        return visited[r][c];
    }

    
    
    //#######  TRIES  #######//
    // 208. Implement Trie (Prefix Tree) (https://leetcode.com/problems/implement-trie-prefix-tree/description/) - Medium
    class Trie {
        Map<Character, Trie> children;
        boolean isWord;
    
        public Trie() {
            this.children = new HashMap<>();
            this.isWord = false;
        }
        
        public void insert(String word) {
            Trie current = this;
            for (char c: word.toCharArray()) {
                current.children.putIfAbsent(c, new Trie());
                current = current.children.get(c);
            }
            current.isWord = true;
        }
        
        public boolean search(String word) {
            Trie current = this;
            for (char c: word.toCharArray()) {
                if (!current.children.containsKey(c)) {
                    return false;
                }
                current = current.children.get(c);
            }
            return current.isWord;
        }
        
        public boolean startsWith(String prefix) {
            Trie current = this;
            for (char c: prefix.toCharArray()) {
                if (!current.children.containsKey(c)) {
                    return false;
                }
                current = current.children.get(c);
            }
            return true;
        }
    }
    /**
     * Your Trie object will be instantiated and called as such:
     * Trie obj = new Trie();
     * obj.insert(word);
     * boolean param_2 = obj.search(word);
     * boolean param_3 = obj.startsWith(prefix);
     */
    


    
    //#######  GRAPHS  #######//
    // 200. Number of Islands (https://leetcode.com/problems/number-of-islands/description/) - Medium
    char[][] grid;

    public int markIslands(int a, int b) {
        if (a < 0 || b < 0 || a >= grid.length || b >= grid[0].length || this.grid[a][b] == '0') {
            return 0;
        }
        this.grid[a][b] = '0';
        markIslands(a-1, b);
        markIslands(a+1, b);
        markIslands(a, b-1);
        markIslands(a, b+1);
        return 1;
    }

    public int numIslands(char[][] grid) {
        int count = 0;
        this.grid = grid;
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                count += markIslands(r, c);
            }
        }
        return count;
    }
    



    //#######  SUBSETS  #######//
    // 78. Subsets (https://leetcode.com/problems/subsets/description/) - Medium
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> output = new ArrayList<>();
        output.add(new ArrayList<Integer>());
        for(Integer n: nums) {
            int l = output.size();
            for(int i=0; i<l; i++) {
                List<Integer> newList = new ArrayList<>(output.get(i));
                newList.add(n);
                output.add(newList);
            }
        }
        return output;
    }
    
}




class Main {
    public static void main(String[] args) {
        //  ArrayList<List<String>> questions = new ArrayList<List<String>>();
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
