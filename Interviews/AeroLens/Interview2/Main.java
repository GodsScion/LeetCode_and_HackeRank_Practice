package Interviews.AeroLens.Interview2;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

// Definition of the Student class
class Student {
    private String name;
    private String grade;
    private int score;

    // Constructor
    public Student(String name, String grade, int score) {
        this.name = name;
        this.grade = grade;
        this.score = score;
    }

    // Getters
    public String getName() {
        return name;
    }

    public String getGrade() {
        return grade;
    }

    public int getScore() {
        return score;
    }

    // toString method for easy printing of Student objects
    @Override
    public String toString() {
        return "Student{" +
                "name='" + name + '\'' +
                ", grade='" + grade + '\'' +
                ", score=" + score +
                '}';
    }

    // Utility method to provide a list of default students
    public static List<Student> getDefaultStudents() {
        List<Student> students = new ArrayList<>();
        students.add(new Student("Alice", "A", 88));
        students.add(new Student("Bob", "B", 45));
        students.add(new Student("Carol", "A", 67));
        students.add(new Student("Dean", "A", 90));
        students.add(new Student("Eve", "B", 72));
        return students;
    }
}

public class Main {
    public static void main(String[] args) {
        List<Student> students = Student.getDefaultStudents();

        // //Question:
        // Given a list of student objects (each student has properties: "name", "grade", and "score"), use the Java Stream API to:
        // Filter out students who scored less than 50.
        // Sort remaining students by their score in descending order.
        // Collect the top 3 students' names and scores.
        // Example:
        // Input:
        // students: [
        // {"name": "Alice", "grade": "A", "score": 88},
        // {"name": "Bob", "grade": "B", "score": 45},
        // {"name": "Carol", "grade": "A", "score": 67},
        // {"name": "Dean", "grade": "A", "score": 90},
        // {"name": "Eve", "grade": "B", "score": 72}
        // ]
        // Output:
        // Return [("Dean", 90), ("Alice", 88), ("Eve", 72)] // List of top 3 students who scored above 50, sorted by score descending

        // Stream operations to process the list of students 
        List<String> topStudents = students.stream()
                .filter(student -> student.getScore() > 50)
                .sorted(Comparator.comparingInt(Student::getScore).reversed())
                .limit(3).map(student -> String.format("(%s, %d)", student.getName(), student.getScore()))
                .collect(Collectors.toList());


        // Print the top students
        topStudents.forEach(System.out::println);
    }
}
