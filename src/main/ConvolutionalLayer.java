package main;

import java.util.Arrays;
import java.util.function.Supplier;

public class ConvolutionalLayer extends Layer {

    private final double[][][] kernels;

    private final double[][][] kernelsVelocity;

    private final double[][][] kernelsVelocitySquared;

    private final double[][][] kernelGradient;

    private final int inputWidth, inputHeight, inputLength;
    private final int kernelWidth, kernelHeight, numKernels;
    private final int strideWidth, strideHeight;
    private final int paddingWidth;
    private final int paddingHeight;
    private final boolean padding;
    private final int[][][] inputVectorToInputMatrix;


    public ConvolutionalLayer(int inputWidth, int inputHeight, int inputLength,
                              int kernelWidth, int kernelHeight, int numKernels,
                              int strideWidth, int strideHeight, Supplier<Double> initializer) {
        this(inputWidth, inputHeight, inputLength, kernelWidth, kernelHeight, numKernels, strideWidth, strideHeight, false, initializer);
    }

    public ConvolutionalLayer(int inputWidth, int inputHeight, int inputLength,
                              int kernelWidth, int kernelHeight, int numKernels,
                              int strideWidth, int strideHeight, boolean padding, Supplier<Double> initializer) {
        super(padding ? inputWidth * inputHeight * inputLength * numKernels :
                (Math.ceilDiv(inputWidth - kernelWidth + 1, strideWidth) + 1) *
                        (Math.ceilDiv(inputHeight - kernelHeight + 1, strideHeight) + 1) *
                        inputLength, initializer);
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputLength = inputLength;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.numKernels = numKernels;
        this.strideWidth = strideWidth;
        this.strideHeight = strideHeight;
        this.padding = padding;
        if (padding) {
            this.paddingWidth = inputWidth * strideWidth - strideWidth - inputWidth + kernelWidth;
            this.paddingHeight = inputHeight * strideHeight - strideHeight - inputHeight + kernelHeight;
        } else {
            this.paddingWidth = Math.ceilDiv(inputWidth - kernelWidth + 1, strideWidth) * strideWidth + kernelWidth - inputWidth;
            this.paddingHeight = Math.ceilDiv(inputHeight - kernelHeight + 1, strideHeight) * strideHeight + kernelHeight - inputHeight;
        }

        this.kernels = new double[kernelWidth][kernelHeight][numKernels];
        this.kernelsVelocity = new double[kernelWidth][kernelHeight][numKernels];
        this.kernelsVelocitySquared = new double[kernelWidth][kernelHeight][numKernels];
        this.kernelGradient = new double[kernelWidth][kernelHeight][numKernels];

        for (int i = 0; i < kernelWidth; i++)
            for (int j = 0; j < kernelHeight; j++)
                for (int k = 0; k < numKernels; k++)
                    kernels[i][j][k] = initializer.get();

        //initialize inputVectorToInputMatrix converter and find padding
        inputVectorToInputMatrix = new int[inputWidth + paddingWidth][inputHeight + paddingHeight][inputLength];
        int paddingLeft, paddingUp;
        if (padding) { //preserve original dimension size
            paddingLeft = Math.ceilDiv(paddingWidth, 2);
            paddingUp = Math.ceilDiv(paddingHeight, 2);
        } else { //only pad when stride is too big
            paddingLeft = paddingWidth;
            paddingUp = paddingHeight;
        }

        //transform 1D input array into 3D input matrix and add padding
        for (int layer = 0; layer < inputLength; layer++)
            for (int x = 0; x < inputWidth + paddingWidth; x++) {
                int i;
                if (x < paddingLeft) i = paddingLeft - x;
                else if (x >= inputWidth + paddingLeft) i = 2 * inputWidth + paddingLeft - x;
                else i = x - paddingLeft;
                for (int y = 0; y < inputHeight + paddingHeight; y++) {
                    int j;
                    if (x < paddingUp) j = paddingUp - x;
                    else if (x >= inputWidth + paddingUp) j = 2 * inputWidth + paddingUp - x;
                    else j = x - paddingUp;
                    inputVectorToInputMatrix[x][y][layer] = inputWidth * inputHeight * layer + inputWidth * j + i;
                }
            }
    }

    private double getMatrixElementFromVector(double[] inputVector, int x, int y, int layer) {
        return inputVector[inputVectorToInputMatrix[x][y][layer]];
    }

    /** Applies the weights and biases of this java.Layer to the given input. Returns a new array. */
    public double[] calculateWeightedOutput(double[] input) {
        assert inputWidth * inputHeight * inputLength == input.length;

        //use kernels to scan through each layer of input matrix, create output matrix
        int outputWidth = (inputWidth + paddingWidth - kernelWidth) / strideWidth + 1, outputHeight = (inputHeight + paddingHeight - kernelHeight) / strideHeight + 1;
        double[] output = new double[outputWidth * outputHeight * inputLength];
        for (int layer = 0; layer < inputLength; layer++)
            for (int x = 0; x < outputWidth; x++)
                for (int y = 0; y < outputHeight; y++) {
                    //loop kernel through each kernel-region to completely populate a location in the output
                    double weightedSum = 0;
                    for (int scanX = 0; scanX < kernelWidth; scanX++)
                        for (int scanY = 0; scanY < kernelHeight; scanY++)
                            for (int kernel = 0; kernel < numKernels; kernel++)
                                weightedSum += kernels[scanX][scanY][kernel] * getMatrixElementFromVector(input, x * strideWidth + scanX, y * strideHeight + scanY, layer);

                    output[x + y * outputWidth + layer * outputWidth * outputHeight] = weightedSum + bias[y * output.length + x];
                }

        return output;
    }

    /**
     * Given the derivative array of the latest input sum,
     * calculates and shifts the given weight and bias gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    @Override
    public double[] updateGradient(double[] dz_dC, double[] x) {
        double[] da_dC = new double[inputWidth * inputHeight * inputLength];
        int outputWidth = (inputWidth + paddingWidth - kernelWidth) / strideWidth + 1, outputHeight = (inputHeight + paddingHeight - kernelHeight) / strideHeight + 1;
        for (int layer = 0; layer < inputLength; layer++)
            for (int i = 0; i < outputWidth; i++)
                for (int j = 0; j < outputHeight; j++) {
                    int index = i + j * outputWidth + layer * outputWidth * outputHeight;
                    assert Double.isFinite(dz_dC[index]);
                    for (int kernelX = 0; kernelX < kernelWidth; kernelX++)
                        for (int kernelY = 0; kernelY < kernelHeight; kernelY++)
                            for (int kernelZ = 0; kernelZ < numKernels; kernelZ++) {
                                assert Double.isFinite(kernelGradient[kernelX][kernelY][kernelZ]);
                                assert Double.isFinite(getMatrixElementFromVector(x, i * strideWidth + kernelX, j * strideHeight + kernelY, layer));

                                kernelGradient[kernelX][kernelY][kernelZ] += dz_dC[index] * getMatrixElementFromVector(x, i * strideWidth + kernelX, j * strideHeight + kernelY, layer);
                                da_dC[inputVectorToInputMatrix[i * strideWidth + kernelX][j * strideHeight + kernelY][layer]] += dz_dC[index] * kernels[kernelX][kernelY][kernelZ];
                            }
                }

        return da_dC;
    }

    /**
     * Applies the {@code weightGradient} and {@code biasGradient} to the weight and bias of this java.Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    @Override
    public void applyGradiant(double adjustedLearningRate, double momentum, double beta, double epsilon) {
        double correctionMomentum = 1 - Math.pow(momentum, t);
        double correctionBeta = 1 - Math.pow(beta, t);
        for (int x = 0; x < kernelWidth; x++)
            for (int y = 0; y < kernelHeight; y++)
                for (int layer = 0; layer < numKernels; layer++) {
                    assert Double.isFinite(kernelGradient[x][y][layer]);
                    kernelsVelocity[x][y][layer] = momentum * kernelsVelocity[x][y][layer] + (1 - momentum) * kernelGradient[x][y][layer];
                    kernelsVelocitySquared[x][y][layer] = beta * kernelsVelocitySquared[x][y][layer] + (1 - beta) * kernelGradient[x][y][layer] * kernelGradient[x][y][layer];
                    double correctedVelocity = kernelsVelocity[x][y][layer] / correctionMomentum;
                    double correctedVelocitySquared = kernelsVelocitySquared[x][y][layer] / correctionBeta;
                    kernels[x][y][layer] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
                    assert Double.isFinite(kernels[x][y][layer]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nweightsVelocity: " + kernelsVelocity[x][y][layer] + "\nweightsVelocitySquared: " + kernelsVelocitySquared[x][y][layer];
                }
        for (int i = 0; i < bias.length; i++) {
            assert Double.isFinite(biasGradient[i]);
            biasVelocity[i] = momentum * biasVelocity[i] + (1 - momentum) * biasGradient[i];
            biasVelocitySquared[i] = beta * biasVelocitySquared[i] + (1 - beta) * biasGradient[i] * biasGradient[i];
            double correctedVelocity = biasVelocity[i] / correctionMomentum;
            double correctedVelocitySquared = biasVelocitySquared[i] / correctionBeta;
            bias[i] -= adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon);
            assert Double.isFinite(bias[i]) : "\ncorrectedVelocity: " + correctedVelocity + "\ncorrectedVelocitySquared: " + correctedVelocitySquared + "\nbiasVelocity: " + biasVelocity[i] + "\nbiasVelocitySquared: " + biasVelocitySquared[i];
        }
        t++;
    }

    @Override
    public void clearGradient() {
        for (int i = 0; i < kernels.length; i++)
            kernelGradient[i] = new double[kernels[0].length][kernels[0][0].length];
        Arrays.fill(biasGradient, 0);
    }
}
