import java.util.*;

class Solution {
    //#######  ARRAYS AND HASHING  #######//
    // 217
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

    // 242
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

    // 1
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

    // 49
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String,List<String>> hash_map = new HashMap<>();
        for (String word: strs) {
            char[] id = word.toCharArray();
            Arrays.sort(id);
            String idword = new String(id);

            if (!hash_map.containsKey(idword)) {
                hash_map.put(idword, new ArrayList<String>());
            }
            hash_map.get(idword).add(word);
        }

        return new ArrayList<>(hash_map.values());
    }
    // 49
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
    // 49
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

    // 347
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
    // 347
    public int[] topKFrequent2(int[] nums, int k) {
        List<Integer>[] buckets = new List[nums.length + 1];
        Map<Integer, Integer> freq = new HashMap<>();

        for (int num: nums) {
            freq.put(num, freq.getOrDefault(num,0) + 1);
        }

        for (int num: freq.keySet()) {
            int count = freq.get(num);
            if (buckets[count] == null) {
                buckets[count] = new ArrayList<Integer>();
            }
            buckets[count].add(num);
        }

        int[] output = new int[k];
        for (int i = buckets.length-1; i >= 0 ; i--) {
            if(buckets[i] != null) {
                for(int num: buckets[i]) {
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

    // 238
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
    // 238
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

    // 128
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
    // 128
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
    // 125
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

    // 15
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

    // 11
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
    // 121
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

    // 3
    public int lengthOfLongestSubstring(String s) {
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
    // 3
    public int lengthOfLongestSubstring2(String s) {
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

    // 424
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

    // 76
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

    
    //#######  STACK  #######//
    // 20
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

    //#######  BINARY SEARCH  #######//
    // 153
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
    
    // 33
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
    // 206
    /**
    * Definition for singly-linked list. */
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    /* def end */

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

    // 21
    // Same def as 206
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

    // 143
    // Same def as 206
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

    // 19
    // Same def as 206
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

    // 141
    // Same def as 206
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

    // 23. Merge k Sorted Lists (https://leetcode.com/problems/merge-k-sorted-lists/description/)
    // Same def as 206
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

    // 23. Merge k Sorted Lists (https://leetcode.com/problems/merge-k-sorted-lists/description/)
    // Same def as 206
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
    // 226. Invert Binary Tree (https://leetcode.com/problems/invert-binary-tree/description/)
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
