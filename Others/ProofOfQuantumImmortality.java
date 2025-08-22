/*
 * Author: Sai Vignesh Golla
 * License: MIT
 * Email: saivigneshgolla@gmail.com
 * 
 * Disclaimer:
 * This program is provided for educational and theoretical purposes only.
 * The authors and contributors of this program are not liabile for any direct or indirect consequences
 * arising from the interaction, use, misuse or any activity involving directly or indirectly of this program, or related programs, or related topics.
 * Use this program at your own risk.
 * This simulation is purely theoretical and the authors do not endorse any harmful behavior
 * or actions based on the concepts explored in this program.
 * 
 * Copyright (c) 2025 Sai Vignesh Golla
 */


package Others;


import java.math.BigInteger;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

public class ProofOfQuantumImmortality {
    private static final String LOG_FILE_PATH = "simulation_log.txt";

    private static void logToFile(String message) {
        try {
            File logFile = new File(LOG_FILE_PATH);
            if (!logFile.exists()) {
                logFile.createNewFile();
            }
            try (FileWriter fw = new FileWriter(logFile, true)) {
                fw.write(message + System.lineSeparator());
            }
        } catch (IOException e) {
            System.err.println("Failed to write to log file: " + e.getMessage());
        }
    }
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
        long experimentStartTime = System.nanoTime();

        
        long trials = 100; // Number of trials to simulate. Higher, the better for proof, but also more probability for time-consumed and failure.
        BigInteger simulationCount = BigInteger.ZERO;
        

        while (true) {
            long simulationStartTime = System.nanoTime();
            simulationCount = simulationCount.add(BigInteger.ONE);
            boolean survived = true;
            for (int i = 0; i < trials; i++) {
                if (!tryLuck()) {
                    survived = false;
                    String msg = String.format("Simulation %d. Survived %d trials.", simulationCount, i + 1);
                    System.out.println(msg);
                    logToFile(msg);
                    break;
                }
            }
            if (survived) {
                String msg1 = "Survived all trials, this kind of whispers a possibility for quantum immortality!";
                String msg2 = String.format("Total time elapsed for this simulation: %d ns", System.nanoTime() - simulationStartTime);
                System.out.println(msg1);
                System.out.println(msg2);
                logToFile(msg1);
                logToFile(msg2);
                break;
            }
            String msg3 = String.format("Did not survive, restarting simulation. Time elapsed: %d ns", System.nanoTime() - simulationStartTime);
            System.out.println(msg3);
            logToFile(msg3);
        }

    StringBuilder finalMsg = new StringBuilder();
    finalMsg.append("\n#################################################################################\n");
    finalMsg.append("Congratulations! You have successfully simulated a scenario that suggests quantum immortality. Was this worth your time?\n");
    finalMsg.append(String.format("Total simulations run: %d. Total time elapsed: %d ms\n", simulationCount, (System.nanoTime() - experimentStartTime) / 1_000_000));
    finalMsg.append("This simulation is purely theoretical and should not be taken as a real-life proof of quantum immortality.\n");
    finalMsg.append("Always prioritize safety and well-being in real life over theoretical concepts.\n");
    finalMsg.append("Thank you for running this simulation!\n");
    finalMsg.append("Have a great day!\n");
    System.out.println(finalMsg.toString());
    logToFile(finalMsg.toString());
    System.exit(0);
    }
}
