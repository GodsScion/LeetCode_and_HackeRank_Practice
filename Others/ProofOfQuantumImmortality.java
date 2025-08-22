package Others;

public class ProofOfQuantumImmortality {
    public static boolean tryLuck(float survivalProbability) {
        if (survivalProbability <= 0.0 || survivalProbability >= 1.0) {
            throw new IllegalArgumentException("Survival probability must be between 0 and 1.");
        }
        return Math.random() > survivalProbability;
    }

    public static boolean tryLuck() {
        return Math.random() > 0.8;
    }

    public static void main(String[] args) {
        long trials = 100; // Number of trials to simulate. Higher, the better for proof, but also more probability for time-consumed and failure.
        long simulationCount = 0; // Use of BigInteger is recommended, used primitive for performance.
        long experimentStartTime = System.nanoTime();

        while (true) {
            long simulationStartTime = System.nanoTime();
            simulationCount++;
            boolean survived = true;
            for (int i = 0; i < trials; i++) {
                if (!tryLuck()) {
                    survived = false;
                    System.out.println(String.format("Simulation %d. Survived %d trials.", simulationCount, i + 1));
                    break;
                }
            }
            if (survived) {
                System.out.println("Survived all trials, this kind of whispers a possibility for quantum immortality!");
                System.out.println(String.format("Total time elapsed for this simulation: %d ms", (System.nanoTime() - simulationStartTime) / 1_000_000));
                break;
            }
            System.out.println(String.format("Did not survive, restarting simulation. Time elapsed: %d ms", (System.nanoTime() - simulationStartTime) / 1_000_000));
        }

        System.out.println("\n#################################################################################\n");
        System.out.println("Congratulations! You have successfully simulated a scenario that suggests quantum immortality. Was this worth your time?");
        System.out.println(String.format("Total simulations run: %d. Total time elapsed: %d ms", simulationCount, (System.nanoTime() - experimentStartTime) / 1_000_000));
        System.out.println("This simulation is purely theoretical and should not be taken as a real-life proof of quantum immortality.");
        System.out.println("Always prioritize safety and well-being in real life over theoretical concepts.");
        System.out.println("Thank you for running this simulation!");
        System.out.println("Have a great day!");
        System.exit(0);
    }
}
