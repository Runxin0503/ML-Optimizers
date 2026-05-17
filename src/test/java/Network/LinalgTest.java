package Network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

/**
 * Edge-case unit tests for {@link Linalg}, the static linear-algebra helpers.
 *
 * <p>Each test pins down one specific behaviour: a known result, an empty input, a dimension
 * mismatch, or a numeric extreme. Tests that expect an {@link AssertionError} rely on Java
 * assertions being enabled ({@code -ea}); Maven Surefire enables them by default.
 */
class LinalgTest {

    private static final double DELTA = 1e-12;

    // ---- matrixMultiply ----------------------------------------------------

    @Test
    void matrixMultiply_knownResult() {
        // 'input' indexes the rows of 'matrix': output[j] = sum_i(input[i] * matrix[i][j])
        double[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        double[] input = {1, 1};
        assertArrayEquals(new double[]{5, 7, 9}, Linalg.matrixMultiply(matrix, input), DELTA);
    }

    @Test
    void matrixMultiply_weightedRows() {
        double[][] matrix = {{1, 0, 0}, {0, 1, 0}};
        double[] input = {2, 3};
        assertArrayEquals(new double[]{2, 3, 0}, Linalg.matrixMultiply(matrix, input), DELTA);
    }

    @Test
    void matrixMultiply_singleElement() {
        assertArrayEquals(new double[]{10},
                Linalg.matrixMultiply(new double[][]{{2}}, new double[]{5}), DELTA);
    }

    @Test
    void matrixMultiply_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.matrixMultiply(new double[][]{{1, 2}}, new double[]{1, 2}));
    }

    @Test
    void matrixMultiply_emptyMatrixAndVector_throwsArrayIndexOutOfBounds() {
        // the collector's supplier dereferences matrix[0] even when the stream is empty
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> Linalg.matrixMultiply(new double[0][], new double[0]));
    }

    @Test
    void matrixMultiply_nanPropagates() {
        // Linalg has no finite-value guard, so NaN flows straight through to the output
        double[] result = Linalg.matrixMultiply(new double[][]{{1.0}}, new double[]{Double.NaN});
        assertTrue(Double.isNaN(result[0]));
    }

    @Test
    void matrixMultiply_identityMatrix_returnsInputUnchanged() {
        double[][] identity = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        double[] input = {3, -2, 1};
        double[] out = Linalg.matrixMultiply(identity, input);
        assertArrayEquals(input, out, DELTA);
    }

    @Test
    void matrixMultiply_zeroMatrix_returnsZeros() {
        double[][] zero = {{0, 0}, {0, 0}};
        double[] out = Linalg.matrixMultiply(zero, new double[]{1, 2});
        assertArrayEquals(new double[]{0, 0}, out, DELTA);
    }

    @Test
    void matrixMultiply_returnsNewArray_notAlias() {
        double[][] m = {{1}};
        double[] in = {5};
        double[] out = Linalg.matrixMultiply(m, in);
        assertNotSame(in, out);
        assertArrayEquals(new double[]{5}, out, DELTA);
    }

    @Test
    void matrixMultiply_outputLengthIsMatrixColumnCount() {
        double[][] m = {{1, 2, 3}}; // 1x3
        double[] out = Linalg.matrixMultiply(m, new double[]{2});
        assertEquals(3, out.length);
    }

    @Test
    void matrixMultiply_infiniteInputValue_propagates() {
        double[] out = Linalg.matrixMultiply(new double[][]{{1.0}}, new double[]{Double.POSITIVE_INFINITY});
        assertTrue(Double.isInfinite(out[0]));
    }

    @Test
    void matrixMultiply_rectangularTallMatrix_knownResult() {
        double[][] m = {{1}, {2}, {3}, {4}}; // 4x1 -> output length 1
        double[] in = {2, 0, 0, 1};
        assertArrayEquals(new double[]{6}, Linalg.matrixMultiply(m, in), DELTA);
    }

    // ---- dotProduct --------------------------------------------------------

    @Test
    void dotProduct_knownResult() {
        assertEquals(32.0, Linalg.dotProduct(new double[]{1, 2, 3}, new double[]{4, 5, 6}), DELTA);
    }

    @Test
    void dotProduct_orthogonalVectors_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[]{1, 0}, new double[]{0, 1}), DELTA);
    }

    @Test
    void dotProduct_emptyArrays_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[0], new double[0]), DELTA);
    }

    @Test
    void dotProduct_singleElement() {
        assertEquals(12.0, Linalg.dotProduct(new double[]{3}, new double[]{4}), DELTA);
    }

    @Test
    void dotProduct_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.dotProduct(new double[]{1}, new double[]{1, 2}));
    }

    @Test
    void dotProduct_overflowToInfinity() {
        // no overflow handling: 1e200 * 1e200 = 1e400, which is beyond Double.MAX_VALUE
        assertTrue(Double.isInfinite(Linalg.dotProduct(new double[]{1e200}, new double[]{1e200})));
    }

    @Test
    void dotProduct_bothNegativeVectors_isPositive() {
        // (-1)*(-1) + (-2)*(-3) = 1 + 6 = 7
        assertEquals(7.0, Linalg.dotProduct(new double[]{-1, -2}, new double[]{-1, -3}), DELTA);
    }

    @Test
    void dotProduct_oppositeSignVectors_isNegative() {
        assertEquals(-8.0, Linalg.dotProduct(new double[]{2, 2}, new double[]{-3, -1}), DELTA);
    }

    @Test
    void dotProduct_allZeros_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[]{0, 0, 0}, new double[]{0, 0, 0}), DELTA);
    }

    @Test
    void dotProduct_largeVector_matchesManualSum() {
        int n = 1000;
        double[] a = new double[n];
        double[] b = new double[n];
        double manual = 0.0;
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.5;
            b[i] = i * 0.25;
            manual += a[i] * b[i];
        }
        assertEquals(manual, Linalg.dotProduct(a, b), 1e-9);
    }

    @Test
    void dotProduct_underflowSmallNumbers_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[]{1e-300}, new double[]{1e-300}), DELTA);
    }

    // ---- multiply ----------------------------------------------------------

    @Test
    void multiply_knownResult() {
        assertArrayEquals(new double[]{4, 10, 18},
                Linalg.multiply(new double[]{1, 2, 3}, new double[]{4, 5, 6}), DELTA);
    }

    @Test
    void multiply_emptyArrays_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.multiply(new double[0], new double[0]), DELTA);
    }

    @Test
    void multiply_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.multiply(new double[]{1, 2}, new double[]{1}));
    }

    @Test
    void multiply_onesVector_isIdentity() {
        double[] v = {3, 4, 5};
        double[] ones = {1, 1, 1};
        assertArrayEquals(v, Linalg.multiply(v, ones), DELTA);
    }

    @Test
    void multiply_byZeroVector_isZeros() {
        double[] v = {3, -4};
        assertArrayEquals(new double[]{0, 0}, Linalg.multiply(v, new double[]{0, 0}), DELTA);
    }

    @Test
    void multiply_nanInOneArgument_propagates() {
        double[] res = Linalg.multiply(new double[]{1, Double.NaN}, new double[]{2, 3});
        assertTrue(Double.isNaN(res[1]));
    }

    @Test
    void multiply_returnsNewArray_notAlias() {
        double[] a = {2};
        double[] b = {3};
        double[] out = Linalg.multiply(a, b);
        assertNotSame(a, out);
    }

    // ---- scale / scaleInPlace ---------------------------------------------

    @Test
    void scale_knownResult() {
        assertArrayEquals(new double[]{2, 4, 6}, Linalg.scale(2, new double[]{1, 2, 3}), DELTA);
    }

    @Test
    void scale_byZero_returnsZeros() {
        assertArrayEquals(new double[]{0, 0, 0}, Linalg.scale(0, new double[]{1, -2, 3}), DELTA);
    }

    @Test
    void scale_byNegativeOne_negates() {
        assertArrayEquals(new double[]{-1, 2, -3}, Linalg.scale(-1, new double[]{1, -2, 3}), DELTA);
    }

    @Test
    void scale_byNaN_producesNaNArray() {
        // scale has no finite-value guard
        double[] result = Linalg.scale(Double.NaN, new double[]{1, 2});
        assertTrue(Double.isNaN(result[0]) && Double.isNaN(result[1]));
    }

    @Test
    void scale_emptyArray_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.scale(5, new double[0]), DELTA);
    }

    @Test
    void scaleInPlace_mutatesArgument() {
        double[] array = {1, 2, 3};
        Linalg.scaleInPlace(3, array);
        assertArrayEquals(new double[]{3, 6, 9}, array, DELTA);
    }

    @Test
    void scaleInPlace_emptyArray_isNoOp() {
        double[] array = new double[0];
        Linalg.scaleInPlace(2, array);
        assertEquals(0, array.length);
    }

    @Test
    void scale_byLargeConstant_overflowsToInfinity() {
        double[] out = Linalg.scale(1e308, new double[]{2});
        assertTrue(Double.isInfinite(out[0]));
    }

    @Test
    void scale_byInfiniteConstant_producesInfinityArray() {
        double[] out = Linalg.scale(Double.POSITIVE_INFINITY, new double[]{1, 2});
        assertTrue(Double.isInfinite(out[0]) && Double.isInfinite(out[1]));
    }

    @Test
    void scaleInPlace_byNegative_negatesAndMutates() {
        double[] a = {1, -2};
        Linalg.scaleInPlace(-1, a);
        assertArrayEquals(new double[]{-1, 2}, a, DELTA);
    }

    @Test
    void scaleInPlace_byNaN_makesArrayNaN() {
        double[] a = {1, 2};
        Linalg.scaleInPlace(Double.NaN, a);
        assertTrue(Double.isNaN(a[0]) && Double.isNaN(a[1]));
    }

    @Test
    void scale_returnsNewArray_notAlias() {
        double[] src = {1, 2};
        double[] out = Linalg.scale(3, src);
        assertNotSame(src, out);
        assertArrayEquals(new double[]{1, 2}, src, DELTA);
    }

    // ---- add / addInPlace --------------------------------------------------

    @Test
    void add_knownResult() {
        assertArrayEquals(new double[]{4, 6}, Linalg.add(new double[]{1, 2}, new double[]{3, 4}), DELTA);
    }

    @Test
    void add_emptyArrays_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.add(new double[0], new double[0]), DELTA);
    }

    @Test
    void add_exactCancellation() {
        assertArrayEquals(new double[]{0.0}, Linalg.add(new double[]{1e100}, new double[]{-1e100}), DELTA);
    }

    @Test
    void add_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.add(new double[]{1}, new double[]{1, 2}));
    }

    @Test
    void addInPlace_mutatesFirstArgument() {
        double[] first = {1, 2, 3};
        Linalg.addInPlace(first, new double[]{10, 20, 30});
        assertArrayEquals(new double[]{11, 22, 33}, first, DELTA);
    }

    @Test
    void addInPlace_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.addInPlace(new double[]{1, 2}, new double[]{1}));
    }

    @Test
    void add_infinityPlusFinite_isInfinity() {
        double[] res = Linalg.add(new double[]{Double.POSITIVE_INFINITY}, new double[]{1.0});
        assertTrue(Double.isInfinite(res[0]));
    }

    @Test
    void add_infinityPlusNegativeInfinity_isNaN() {
        double[] res = Linalg.add(new double[]{Double.POSITIVE_INFINITY}, new double[]{Double.NEGATIVE_INFINITY});
        assertTrue(Double.isNaN(res[0]));
    }

    @Test
    void add_singleElement_knownResult() {
        assertArrayEquals(new double[]{7}, Linalg.add(new double[]{3}, new double[]{4}), DELTA);
    }

    @Test
    void addInPlace_arrayWithItself_doubles() {
        double[] a = {2, 3};
        Linalg.addInPlace(a, a);
        assertArrayEquals(new double[]{4, 6}, a, DELTA);
    }

    @Test
    void addInPlace_emptyArrays_isNoOp() {
        double[] a = new double[0];
        Linalg.addInPlace(a, new double[0]);
        assertEquals(0, a.length);
    }

    // ---- sum ---------------------------------------------------------------

    @Test
    void sum_knownResult() {
        assertEquals(6.0, Linalg.sum(new double[]{1, 2, 3}), DELTA);
    }

    @Test
    void sum_emptyArray_isZero() {
        assertEquals(0.0, Linalg.sum(new double[0]), DELTA);
    }

    @Test
    void sum_singleElement() {
        assertEquals(-7.0, Linalg.sum(new double[]{-7}), DELTA);
    }

    @Test
    void sum_withNaN_isNaN() {
        assertTrue(Double.isNaN(Linalg.sum(new double[]{1, Double.NaN, 3})));
    }

    @Test
    void sum_allNegative_isCorrectNegative() {
        assertEquals(-6.0, Linalg.sum(new double[]{-1, -2, -3}), DELTA);
    }

    @Test
    void sum_withPositiveInfinity_isInfinity() {
        assertTrue(Double.isInfinite(Linalg.sum(new double[]{1, Double.POSITIVE_INFINITY}))); 
    }

    @Test
    void sum_withNegativeInfinity_isNegativeInfinity() {
        assertTrue(Double.isInfinite(Linalg.sum(new double[]{-1, Double.NEGATIVE_INFINITY}))); 
    }

    @Test
    void sum_largeArrayOfOnes_isCount() {
        int n = 10000;
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = 1.0;
        assertEquals(n, Linalg.sum(a), DELTA);
    }

    @Test
    void sum_mixedSignsCancelsToZero() {
        assertEquals(0.0, Linalg.sum(new double[]{1, -1, 2, -2}), DELTA);
    }
}
