package Network;

import java.util.Arrays;
import java.util.stream.IntStream;

public class Linalg {
    public static double[] matrixMultiply(double[][] matrix, double[] input) {
        assert matrix.length == input.length;
//        double[] output = new double[matrix[0].length];
//        for (int i = 0; i < matrix.length; i++)
//            output = Linalg.add(output, Linalg.scale(input[i], matrix[i]));
        return IntStream.range(0, input.length).parallel().mapToObj(i -> Linalg.scale(input[i], matrix[i]))
                .collect(() -> new double[matrix[0].length], Linalg::addInPlace, Linalg::addInPlace);
    }

    public static double dotProduct(double[] first, double[] second) {
        assert first.length == second.length;
        return IntStream.range(0, first.length).mapToDouble(i -> first[i] * second[i]).sum();
    }

    public static double[] multiply(double[] first, double[] second) {
        assert first.length == second.length;
        return IntStream.range(0, first.length).mapToDouble(i -> first[i] * second[i]).toArray();
    }

    public static double[] scale(double constant, double[] array) {
        return Arrays.stream(array).parallel().map(v -> constant * v).toArray();
    }

    public static void scaleInPlace(double constant, double[] array) {
        IntStream.range(0,array.length).parallel().forEach(i -> array[i] *= constant);
    }

    public static double[] add(double[] first, double[] second) {
        assert first.length == second.length;
        return IntStream.range(0, first.length).mapToDouble(i -> first[i] + second[i]).toArray();
    }

    public static void addInPlace(double[] first, double[] second) {
        assert first.length == second.length;
        IntStream.range(0, first.length).parallel().forEach(i -> first[i] += second[i]);
    }

    public static double sum(double[] arr) {
        return Arrays.stream(arr).sum();
    }
}
