package Interviews.AeroLens.Interview1;
public class InterviewClass {
    public String occurrence(String s) {
        StringBuilder sb = new StringBuilder();
        
        int count = 0;
        char prevChar = s.charAt(0);
        for (char c: s.toCharArray()) {
            if (c == prevChar) {
                count++;
            } else {
                sb.append(prevChar);
                sb.append(count);
                prevChar = c;
                count = 1;
            }
        }
        sb.append(prevChar);
        sb.append(count);
        return sb.toString();
    } 
}

class Main {
    public static void main(String[] args) {
        InterviewClass ic = new InterviewClass();
        System.out.println(ic.occurrence("aaaabbccc"));
    }
}
