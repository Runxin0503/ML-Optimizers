package Network;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Edge-case unit tests for {@link ConvolutionalLayer} (a package-private class, hence the
 * {@code Network} package declaration).
 *
 * <p>{@code kernels} is {@code private}, so the suite uses freshly-constructed layers (all-zero
 * kernels and bias) and drives kernel state through {@code updateGradient} -> {@code applyGradient},
 * observing the effect through {@code calculateWeightedOutput}. To keep the convolution arithmetic
 * hand-checkable despite the internal reflection-padding index map, the smaller tests use a
 * 2x2 input with a single 2x2 kernel and uniform inputs (so every kernel weight multiplies 1.0).
 * Tests that expect an {@link AssertionError} rely on {@code -ea} (Maven Surefire default).
 */
class ConvolutionalLayerTest {

    private static final double DELTA = 1e-12;

    // ---- constructor: output-dimension math --------------------------------

    @Test
    void nodes_noPadding_followCeilDivFormula() {
        // outputW = ceilDiv(7-3+1, 2) = 3, outputH = 3, nodes = 3 * 3 * numKernels(4) = 36
        ConvolutionalLayer layer = new ConvolutionalLayer(7, 7, 1, 3, 3, 4, 2, 2, false);
        assertEquals(36, layer.nodes);
    }

    @Test
    void nodes_withPadding_preserveInputArea() {
        // with padding the layer keeps the input area: nodes = inputW * inputH * numKernels
        ConvolutionalLayer layer = new ConvolutionalLayer(5, 5, 1, 3, 3, 2, 1, 1, true);
        assertEquals(5 * 5 * 2, layer.nodes);
    }

    @Test
    void constructor_kernelWiderThanInput_throwsNegativeArraySize() {
        // kernelWidth(6) > inputWidth(3) makes ceilDiv(3-6+1, 1) = -2, so nodes = -2 * 3 * 1 = -6
        // and the Layer super-constructor allocates `new double[-6]`
        assertThrows(NegativeArraySizeException.class,
                () -> new ConvolutionalLayer(3, 5, 1, 6, 3, 1, 1, 1, false));
    }

    @Test
    void getNumParameters_isKernelsPlusBiases() {
        // numKernels * kernelW * kernelH + nodes  =  4 * 3 * 3 + 36  =  72
        ConvolutionalLayer layer = new ConvolutionalLayer(7, 7, 1, 3, 3, 4, 2, 2, false);
        assertEquals(72, layer.getNumParameters());
    }

    // ---- calculateWeightedOutput ------------------------------------------

    @Test
    void calculateWeightedOutput_freshLayerProducesZeroVectorOfNodeCount() {
        ConvolutionalLayer layer = new ConvolutionalLayer(5, 5, 1, 3, 3, 2, 1, 1, false);
        double[] output = layer.calculateWeightedOutput(new double[25]);
        assertEquals(layer.nodes, output.length);
        assertArrayEquals(new double[layer.nodes], output, DELTA);
    }

    @Test
    void calculateWeightedOutput_wrongInputLength_throwsAssertionError() {
        // requires -ea: input length must equal inputWidth * inputHeight * inputLength (25 here)
        ConvolutionalLayer layer = new ConvolutionalLayer(5, 5, 1, 3, 3, 2, 1, 1, false);
        assertThrows(AssertionError.class, () -> layer.calculateWeightedOutput(new double[24]));
    }

    // ---- updateGradient ----------------------------------------------------

    @Test
    void updateGradient_returnsDaDcSizedToTheInputVolume() {
        ConvolutionalLayer layer = new ConvolutionalLayer(5, 5, 1, 3, 3, 2, 1, 1, false);
        double[] daDC = layer.updateGradient(new double[layer.nodes], new double[25]);
        // fresh kernels are zero, so da_dC is a zero vector the size of the input volume
        assertArrayEquals(new double[25], daDC, DELTA);
    }

    // ---- applyGradient: deterministic kernel update ------------------------

    @Test
    void applyGradient_sgd_modifiesKernelsObservably() {
        // 2x2 input, one 2x2 kernel, stride 1, no padding -> a single output node.
        ConvolutionalLayer layer = new ConvolutionalLayer(2, 2, 1, 2, 2, 1, 1, 1, false);
        // dz_dC = [1] and uniform x = [1,1,1,1] -> every kernelsGradient entry becomes 1.0
        layer.updateGradient(new double[]{1.0}, new double[]{1, 1, 1, 1});
        // SGD: each of the 4 kernel weights -> 0 - 0.1 * 1.0 = -0.1
        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        // with uniform input [1,1,1,1] the output is the sum of the 4 kernel weights: 4 * -0.1
        assertArrayEquals(new double[]{-0.4}, layer.calculateWeightedOutput(new double[]{1, 1, 1, 1}), DELTA);
    }

    @Test
    void clearGradient_makesSubsequentApplyGradientANoOp() {
        ConvolutionalLayer layer = new ConvolutionalLayer(2, 2, 1, 2, 2, 1, 1, 1, false);
        layer.updateGradient(new double[]{1.0}, new double[]{1, 1, 1, 1});
        layer.clearGradient();
        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        // gradients were zeroed, so the kernels remain zero
        assertArrayEquals(new double[]{0.0}, layer.calculateWeightedOutput(new double[]{1, 1, 1, 1}), DELTA);
    }

    @Test
    void applyGradient_rmsProp_keepsKernelsFinite() {
        // FAIL-LOUDLY: ConvolutionalLayer.java:200 divides by sqrt(kernelsGradient + epsilon)
        // instead of sqrt(kernelsVelocitySquared + epsilon) (DenseLayer.java:87 does it right).
        // A negative kernel gradient therefore feeds sqrt() a negative number -> NaN kernels.
        ConvolutionalLayer layer = new ConvolutionalLayer(2, 2, 1, 2, 2, 1, 1, 1, false);
        layer.initialize(() -> 0.0, Optimizer.RMS_PROP);
        layer.updateGradient(new double[]{-1.0}, new double[]{1, 1, 1, 1}); // negative gradient
        layer.applyGradient(Optimizer.RMS_PROP, 0.1, 0, 0.9, 1e-8);
        double[] output = layer.calculateWeightedOutput(new double[]{1, 1, 1, 1});
        assertTrue(Double.isFinite(output[0]),
                "convolution output went non-finite after an RMS_PROP update");
    }

    // ---- equals ------------------------------------------------------------

    @Test
    void equals_layerEqualsItself() {
        // FAIL-LOUDLY: ConvolutionalLayer.java:253 reads `... || super.equals(obj)` (a `!` is
        // missing), so when the base fields DO match it returns false. A layer is therefore
        // never equal to itself or to an identical layer.
        ConvolutionalLayer layer = new ConvolutionalLayer(4, 4, 1, 2, 2, 1, 1, 1, false);
        assertTrue(layer.equals(layer));
    }

    // ---- clone -------------------------------------------------------------

    @Test
    void clone_squareKernel_behavesIdenticallyToOriginal() {
        ConvolutionalLayer original = new ConvolutionalLayer(4, 4, 1, 2, 2, 1, 1, 1, false);
        double[] dzDC = new double[original.nodes];
        Arrays.fill(dzDC, 1.0);
        double[] x = new double[16];
        Arrays.fill(x, 1.0);
        original.updateGradient(dzDC, x);
        original.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);

        ConvolutionalLayer copy = (ConvolutionalLayer) original.clone();
        assertArrayEquals(original.calculateWeightedOutput(x), copy.calculateWeightedOutput(x), DELTA);
    }

    @Test
    void clone_nonSquareKernel_behavesIdenticallyToOriginal() {
        // FAIL-LOUDLY: ConvolutionalLayer.java:283 copies `kernels[0].length` (kernelWidth)
        // elements out of each `kernels[i][j]` row, which actually has length kernelHeight.
        // With kernelWidth(3) != kernelHeight(2) the System.arraycopy reads past the row end.
        ConvolutionalLayer original = new ConvolutionalLayer(4, 4, 1, 3, 2, 1, 1, 1, false);
        ConvolutionalLayer copy = (ConvolutionalLayer) original.clone();
        double[] input = new double[16];
        assertArrayEquals(original.calculateWeightedOutput(input),
                copy.calculateWeightedOutput(input), DELTA);
    }

    // ---- toString ----------------------------------------------------------

    @Test
    void toString_mentionsKernelsAndBiases() {
        String text = new ConvolutionalLayer(4, 4, 1, 2, 2, 1, 1, 1, false).toString();
        assertTrue(text.contains("Kernel") && text.contains("Biases"));
    }
}
