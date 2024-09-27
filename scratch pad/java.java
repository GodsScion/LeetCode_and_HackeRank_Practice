import java.util.*;
import java.util.stream.Collectors;


//##### Class ######
// public class T {
//     public int val;

//     public T(int val) {
//         this.val = val;
//     }
// }


class Solution {

    //##### SETs ######
    public void sets() {

        Set<Integer> set = new HashSet<>();
        
        set.size();
        
        set.add(1);

        set.contains(1);

        set.remove(1);

        set.equals(set);

        Iterator<Integer> iterator = set.iterator();
        while(iterator.hasNext()) {
            System.out.println(iterator.next());
        }

        for (Integer i: set) {
            System.out.println(i);
        }

    }
    

    //##### MAPs ######
    public void maps() {
        Map<Integer, String> map = new HashMap<>();

        map.size();

        map.put(1, "One");
        map.putIfAbsent(2, "Two");

        map.containsKey(1);
        map.get(1);
        map.getOrDefault(1, null);

        map.remove(1);
        map.remove(2, "Two");

        map.equals(map);

        Iterator<Integer> iterator = map.keySet().iterator();
        while(iterator.hasNext()) {
            System.out.println(iterator.next());
        }

        for (Map.Entry<Integer, String> entry: map.entrySet()) {
            System.out.println(entry.getKey() + " " + entry.getValue());
        }

    }

/* Maps
 * Sets
 * Stack
 * Queue
 * Lists
 * Priority Queues
 * Streams
*/
    //##### Lists #####
    public void lists() {
        List<Integer> list = new ArrayList<>();

        list.size();

        list.add(0);
        list.addAll(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        list.addAll(list);

        list.contains(1);
        list.indexOf(1);
        list.lastIndexOf(1);
        
        list.get(1); //Index out of bounds if index is out of range
        
        
        
        list.toArray();

    }


    //##### Lists #####
    public void streams() {
        int[] arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        List<Integer> list = new ArrayList<>(Arrays.stream(arr).boxed().collect(Collectors.toList()));

        int total = list.stream().filter(x -> x % 2 != 0).map(x -> x * x).reduce(0, (x, y) -> x + y);
        System.out.println(total);


    }

}

class Main {
    public static void main(String[] args) {
        Solution sol = new Solution();
        sol.streams();
    }
}




