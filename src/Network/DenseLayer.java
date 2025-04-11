package Network;

import Utils.Linalg;
import enums.Optimizer;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/** A single layer of Neurons. Contains fully connected edges to every neuron in the previous layer */
class DenseLayer extends Layer {

    /**
     * A 2D matrix of weights
     * <br>Rows: The neuron {@code n} in the previous layer
     * <br>Columns: Every outgoing synapse from n to this layer's node.
     */
    private final double[][] weights;

    /**
     * A 2D matrix of weight velocities for SGD with momentum
     * <br>Rows: The neuron {@code n} in the previous layer
     * <br>Columns: Every outgoing synapse from n to this layer's node.
     */
    private double[][] weightsVelocity;

    /**
     * A 2D matrix of weight velocities for RMS-Prop
     * <br>Rows: The neuron {@code n} in the previous layer
     * <br>Columns: Every outgoing synapse from n to this layer's node.
     */
    private double[][] weightsVelocitySquared;

    /**
     * A 2D matrix of gradients of the loss function with respect to the weights
     * <br>Rows: The neuron {@code n} in the previous layer
     * <br>Columns: Every outgoing synapse from n to this layer's node.
     */
    private final double[][] weightsGradient;

    DenseLayer(int nodesBefore, int nodes) {
        super(nodes);
        this.weights = new double[nodesBefore][nodes];
        this.weightsGradient = new double[nodesBefore][nodes];
    }

    @Override
    void initialize(Supplier<Double> initializer, Optimizer optimizer) {
        super.initialize(initializer, optimizer);

        if (optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.weightsVelocity = new double[weights.length][nodes];
        if (optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.weightsVelocitySquared = new double[weights.length][nodes];

        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < nodes; j++)
                weights[i][j] = initializer.get();
    }

    @Override
    double[] calculateWeightedOutput(double[] input) {
        return Linalg.add(Linalg.matrixMultiply(weights, input), bias);
    }

    @Override
    double[] updateGradient(double[] dz_dC, double[] x) {
        Linalg.addInPlace(biasGradient, dz_dC);
        double[] result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            Linalg.addInPlace(weightsGradient[i], Linalg.scale(x[i], dz_dC));
            result[i] = Linalg.dotProduct(weights[i], dz_dC);
        }

        return result;
    }

    @Override
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        BiConsumer<Integer, Integer> updateRule;
        switch (optimizer) {
            case SGD -> updateRule = (i, j) -> weights[i][j] -= adjustedLearningRate * weightsGradient[i][j];
            case SGD_MOMENTUM -> updateRule = (i, j) -> {
                weightsVelocity[i][j] = weightsVelocity[i][j] * momentum + (1 - momentum) * weightsGradient[i][j];
                weights[i][j] -= adjustedLearningRate * weightsVelocity[i][j];
            };
            case RMS_PROP -> updateRule = (i, j) -> {
                weightsVelocitySquared[i][j] = beta * weightsVelocitySquared[i][j] + (1 - beta) * (weightsGradient[i][j] * weightsGradient[i][j]);
                weights[i][j] -= adjustedLearningRate * weightsGradient[i][j] / Math.sqrt(weightsVelocitySquared[i][j] + epsilon);
            };
            case ADAM -> {
                double correctionMomentum = 1 - Math.pow(momentum, t);
                double correctionBeta = 1 - Math.pow(beta, t);
                updateRule = (i, j) -> {
                    weightsVelocity[i][j] = momentum * weightsVelocity[i][j] + (1 - momentum) * weightsGradient[i][j];
                    weightsVelocitySquared[i][j] = beta * weightsVelocitySquared[i][j] + (1 - beta) * weightsGradient[i][j] * weightsGradient[i][j];
                    double correctedVelocity = weightsVelocity[i][j] / correctionMomentum;
                    double correctedVelocitySquared = weightsVelocitySquared[i][j] / correctionBeta;
                    weights[i][j] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
                    assert Double.isFinite(weights[i][j]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nweightsVelocity: " + weightsVelocity[i][j] + "\nweightsVelocitySquared: " + weightsVelocitySquared[i][j];
                };
            }
            case null, default -> throw new IllegalStateException("Unexpected value: " + optimizer);
        }

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < nodes; j++) {
                assert Double.isFinite(weightsGradient[i][j]);
                updateRule.accept(i, j);
            }
        }

        super.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
    }

    void clearGradient() {
        for (int i = 0; i < weightsGradient.length; i++) weightsGradient[i] = new double[weights[0].length];
        Arrays.fill(biasGradient, 0);
    }

    @Override
    int getNumParameters() {
        return weights.length * weights[0].length + super.getNumParameters();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Weights: ").append(Arrays.deepToString(weights));
        Layer.ArraysDeepToString(weights, sb);
        sb.append("\nBiases: \n").append(Arrays.toString(bias));
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof DenseLayer o) || super.equals(obj)) return false;
        return Arrays.deepEquals(weights, o.weights) &&
                Arrays.deepEquals(weightsVelocity, o.weightsVelocity) &&
                Arrays.deepEquals(weightsVelocitySquared, o.weightsVelocitySquared) &&
                Arrays.deepEquals(weightsGradient, o.weightsGradient);
    }

    @Override
    public Object clone() {
        int nodesBefore = weights[0].length;
        DenseLayer newLayer = new DenseLayer(nodesBefore, nodes);
        System.arraycopy(bias, 0, newLayer.bias, 0, nodes);
        if (!Objects.isNull(biasVelocity)) {
            newLayer.biasVelocity = new double[biasVelocity.length];
            newLayer.weightsVelocity = new double[weightsVelocity.length][weightsVelocity[0].length];
            System.arraycopy(biasVelocity, 0, newLayer.biasVelocity, 0, nodes);
        }
        if (!Objects.isNull(biasVelocitySquared)) {
            newLayer.biasVelocitySquared = new double[biasVelocitySquared.length];
            newLayer.weightsVelocitySquared = new double[weightsVelocitySquared.length][weightsVelocitySquared[0].length];
            System.arraycopy(biasVelocitySquared, 0, newLayer.biasVelocitySquared, 0, nodes);
        }
        System.arraycopy(biasGradient, 0, newLayer.biasGradient, 0, nodes);
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(weights[i], 0, newLayer.weights[i], 0, nodesBefore);
            if (!Objects.isNull(weightsVelocity))
                System.arraycopy(weightsVelocity[i], 0, newLayer.weightsVelocity[i], 0, nodesBefore);
            if (!Objects.isNull(weightsVelocitySquared))
                System.arraycopy(weightsVelocitySquared[i], 0, newLayer.weightsVelocitySquared[i], 0, nodesBefore);
            System.arraycopy(weightsGradient[i], 0, newLayer.weightsGradient[i], 0, nodesBefore);
        }
        return newLayer;
    }
}
