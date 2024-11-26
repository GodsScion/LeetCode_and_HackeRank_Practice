package Interviews.AeroLens.Interview1;
public class InterviewClass {
    public String occurence(String s) {
        StringBuilder sb = new StringBuilder();
        
        int count = 0;
        char prevchar = s.charAt(0);
        for (char c: s.toCharArray()) {
            if (c == prevchar) {
                count++;
            } else {
                sb.append(prevchar);
                sb.append(count);
                prevchar = c;
                count = 1;
            }
        }
        return sb.toString();
    } 
}

class Main {
    public static void main(String[] args) {
        InterviewClass ic = new InterviewClass();
        System.out.println(ic.occurence("aaaabbccc"));
    }
}
