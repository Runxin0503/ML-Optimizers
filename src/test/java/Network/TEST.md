# Network Test Suite — Documentation

This document catalogues every test in `src/test/java/Network/` and the planned additions.

## Design

- **Pinpoint unit tests** isolate one observable behaviour per method, so a failure names exactly
  what broke. Each new pinpoint test below follows the same one-feature-per-method pattern.
- **Class-wise black-box tests** drive a class through a longer, end-to-end sequence (forward +
  backward, multi-step training, optimizer convergence). They trade pinpoint diagnosis for a
  wider net: if a regression slips past the unit tests, it will usually surface here.
- Most tests rely on `-ea` (enabled by default in Maven Surefire 3.x) because the `Network`
  package uses `assert` as its contract for finite values and dimensions.
- Layer tests use an in-file `StubLayer extends Layer` so `Layer`'s `protected` fields can be
  inspected directly. `weights`/`kernels` are `private`, so subclass tests work via
  `updateGradient` -> `applyGradient` and observe the effect through `calculateWeightedOutput`.

## Test Count Summary

| File                     | Existing | + New Pinpoint | + New Black-Box | Total |
|--------------------------|---------:|---------------:|----------------:|------:|
| LinalgTest               |       32 |             30 |               5 |    67 |
| ActivationTest           |       23 |             25 |               4 |    52 |
| CostTest                 |       13 |             20 |               3 |    36 |
| LayerTest                |       16 |             16 |               3 |    35 |
| DenseLayerTest           |       14 |             16 |               4 |    34 |
| ConvolutionalLayerTest   |       14 |             18 |               4 |    36 |
| NNTest                   |       16 |             20 |               5 |    41 |
| **Total**                |  **128** |        **145** |          **28** |**301**|

---

## LinalgTest

Pure static helpers in [Linalg.java](../../../main/java/Network/Linalg.java).

### Existing Tests (32)

**matrixMultiply** (6)
- `matrixMultiply_knownResult` — 2x3 matrix, len-2 input -> [5, 7, 9]
- `matrixMultiply_weightedRows` — selector rows weighted by input -> [2, 3, 0]
- `matrixMultiply_singleElement` — 1x1 matrix x len-1 input
- `matrixMultiply_dimensionMismatch_throwsAssertionError` — matrix.length != input.length
- `matrixMultiply_emptyMatrixAndVector_throwsArrayIndexOutOfBounds` — collector dereferences `matrix[0]`
- `matrixMultiply_nanPropagates` — NaN in input flows through to output

**dotProduct** (6)
- `dotProduct_knownResult` — [1,2,3]·[4,5,6] = 32
- `dotProduct_orthogonalVectors_isZero` — [1,0]·[0,1] = 0
- `dotProduct_emptyArrays_isZero` — empty stream sum = 0.0
- `dotProduct_singleElement` — [3]·[4] = 12
- `dotProduct_dimensionMismatch_throwsAssertionError`
- `dotProduct_overflowToInfinity` — 1e200 · 1e200 -> Infinity

**multiply** (3)
- `multiply_knownResult` — element-wise [4, 10, 18]
- `multiply_emptyArrays_returnsEmpty`
- `multiply_dimensionMismatch_throwsAssertionError`

**scale / scaleInPlace** (7)
- `scale_knownResult` — 2 * [1,2,3] = [2,4,6]
- `scale_byZero_returnsZeros`
- `scale_byNegativeOne_negates`
- `scale_byNaN_producesNaNArray` — no finite-value guard
- `scale_emptyArray_returnsEmpty`
- `scaleInPlace_mutatesArgument`
- `scaleInPlace_emptyArray_isNoOp`

**add / addInPlace** (6)
- `add_knownResult`
- `add_emptyArrays_returnsEmpty`
- `add_exactCancellation` — 1e100 + -1e100 = 0
- `add_dimensionMismatch_throwsAssertionError`
- `addInPlace_mutatesFirstArgument`
- `addInPlace_dimensionMismatch_throwsAssertionError`

**sum** (4)
- `sum_knownResult` — [1,2,3] = 6
- `sum_emptyArray_isZero`
- `sum_singleElement`
- `sum_withNaN_isNaN`

### Planned Pinpoint Additions (30)

**matrixMultiply** (6)
- `matrixMultiply_identityMatrix_returnsInputUnchanged` — square identity x v = v
- `matrixMultiply_zeroMatrix_returnsZeros`
- `matrixMultiply_returnsNewArray_notAlias` — output is a fresh allocation
- `matrixMultiply_outputLengthIsMatrixColumnCount`
- `matrixMultiply_infiniteInputValue_propagates`
- `matrixMultiply_rectangularTallMatrix_knownResult` — 4x1 matrix variant

**dotProduct** (5)
- `dotProduct_bothNegativeVectors_isPositive`
- `dotProduct_oppositeSignVectors_isNegative`
- `dotProduct_allZeros_isZero`
- `dotProduct_largeVector_matchesManualSum` — len-1000 array
- `dotProduct_underflowSmallNumbers_isZero` — 1e-300 · 1e-300 -> 0.0

**multiply** (4)
- `multiply_onesVector_isIdentity` — multiply by all-ones returns input
- `multiply_byZeroVector_isZeros`
- `multiply_nanInOneArgument_propagates`
- `multiply_returnsNewArray_notAlias`

**scale / scaleInPlace** (5)
- `scale_byLargeConstant_overflowsToInfinity`
- `scale_byInfiniteConstant_producesInfinityArray`
- `scaleInPlace_byNegative_negatesAndMutates`
- `scaleInPlace_byNaN_makesArrayNaN`
- `scale_returnsNewArray_notAlias` — input array untouched

**add / addInPlace** (5)
- `add_infinityPlusFinite_isInfinity`
- `add_infinityPlusNegativeInfinity_isNaN`
- `add_singleElement_knownResult`
- `addInPlace_arrayWithItself_doubles` — `addInPlace(a, a)` works without aliasing bug
- `addInPlace_emptyArrays_isNoOp`

**sum** (5)
- `sum_allNegative_isCorrectNegative`
- `sum_withPositiveInfinity_isInfinity`
- `sum_withNegativeInfinity_isNegativeInfinity`
- `sum_largeArrayOfOnes_isCount` — verify stable sum for len-10_000 ones
- `sum_mixedSignsCancelsToZero`

### Planned Class-Wise Black-Box (5)

- `denseLayerForwardChain_matchesHandComputation` — `add(matrixMultiply(W, x), bias)` chained, hand-checked end-to-end
- `sgdUpdateChain_matchesFormula` — full SGD-momentum chain (`scaleInPlace` + `addInPlace` + `scale` + `addInPlace`) reaches the textbook closed-form value
- `largeArraysAreDeterministic` — repeated parallel calls on a fixed input produce identical results (no parallel-stream nondeterminism)
- `chainedOperationsDoNotMutateOriginals` — sequence of pure helpers (`scale`, `add`, `multiply`) leaves all inputs untouched
- `mixedZeroAndNonZero_invariants` — `sum(scale(0, v)) == 0`, `add(v, scale(-1, v)) == zeros`, `dotProduct(v, zeros) == 0` together

---

## ActivationTest

Activation enum in [Activation.java](../../../main/java/Network/Activation.java).

### Existing Tests (23)

**none** (2)
- `none_calculate_isIdentity`
- `none_derivative_passesGradientThrough`

**ReLU** (2)
- `relu_calculate_clampsAtZeroIncludingBoundary` — input 0 -> 0
- `relu_derivative_isStepFunctionWithZeroAtBoundary`

**sigmoid** (4)
- `sigmoid_calculate_atZeroIsHalf`
- `sigmoid_calculate_saturatesForLargeMagnitude` — ~1 / ~0 at +/-100
- `sigmoid_derivative_atZeroIsQuarter`
- `sigmoid_derivative_saturatedIsZeroAndFinite`

**tanh** (3)
- `tanh_calculate_atZeroIsZero`
- `tanh_calculate_saturatesForLargeMagnitude`
- `tanh_derivative_atZeroIsGradient`

**LeakyReLU** (2)
- `leakyRelu_calculate_leaksNegativesAndZerosBoundary`
- `leakyRelu_derivative_isOneOrTenthSlope`

**softmax** (5)
- `softmax_singleElement_isOne`
- `softmax_equalLogits_isUniform`
- `softmax_largePositiveSpread_isNumericallyStable`
- `softmax_derivative_usesJacobianForm`
- `softmax_allNegativeLogits_shouldReturnValidDistribution` — now passes after the `max`-seed fix

**contract** (3)
- `calculate_withNaNInput_throwsAssertionError`
- `calculate_withInfiniteInput_throwsAssertionError`
- `derivative_withNaNGradient_throwsAssertionError`

**getInitializer** (2)
- `getInitializer_producesFiniteWeightsForNormalSizes`
- `getInitializer_zeroSizes_producesNonFiniteOrThrows`

### Planned Pinpoint Additions (25)

**none** (2)
- `none_calculate_emptyArray_returnsEmpty`
- `none_calculate_returnsNewArray_notAlias`

**ReLU** (3)
- `relu_calculate_largePositive_passesThrough`
- `relu_derivative_largeNegative_isZero`
- `relu_calculate_emptyArray_returnsEmpty`

**sigmoid** (3)
- `sigmoid_calculate_symmetry` — `sigmoid(-x) == 1 - sigmoid(x)`
- `sigmoid_derivative_symmetry` — derivative is even function of `z`
- `sigmoid_calculate_neverNonFinite` — even at +/-1e6, output is a real probability

**tanh** (3)
- `tanh_calculate_symmetry` — `tanh(-x) == -tanh(x)`
- `tanh_derivative_isEven` — `f'(-x) == f'(x)`
- `tanh_calculate_emptyArray_returnsEmpty`

**LeakyReLU** (3)
- `leakyRelu_calculate_veryNegativeInput_scalesByTenth` — -1000 -> -100
- `leakyRelu_calculate_largePositive_passesThrough`
- `leakyRelu_derivative_zeroBoundary_takesNegativeBranch`

**softmax** (6)
- `softmax_calculate_invariantUnderConstantShift` — `softmax(x + c) == softmax(x)`
- `softmax_calculate_outputAlwaysSumsToOne`
- `softmax_calculate_emptyArray_returnsEmpty`
- `softmax_derivative_uniformGradient_isZeroVector` — Jacobian property
- `softmax_derivative_oneHotGradient_matchesFormula`
- `softmax_derivative_returnsArrayOfMatchingLength`

**contract / general** (3)
- `calculate_doesNotMutateInput`
- `derivative_doesNotMutateGradient`
- `derivative_withInfiniteGradient_throwsAssertionError`

**getInitializer** (2)
- `getInitializer_returnsHeForReluAndLeakyRelu_xavierOtherwise` — statistical std-dev check over many draws
- `getInitializer_isDeterministicSupplier` — multiple calls produce different (but finite) values

### Planned Class-Wise Black-Box (4)

- `allActivations_finiteInputProduceFiniteOutput` — sweep each enum constant over a moderate input range
- `allActivations_forwardBackwardChain_isFinite` — apply `calculate` then `derivative`, verify outputs stay finite for many inputs
- `numericalDerivative_matchesAnalytic` — for each activation, central-difference numerical derivative ~ analytic derivative within tolerance
- `softmax_probabilitiesSumToOne_forRandomInputs` — 100 random length-10 inputs, each output sums to 1.0

---

## CostTest

Cost enum in [Cost.java](../../../main/java/Network/Cost.java).

### Existing Tests (13)

**diffSquared** (6)
- `diffSquared_calculate_knownResult`
- `diffSquared_calculate_perfectPredictionIsZero`
- `diffSquared_calculate_singleElement`
- `diffSquared_calculate_emptyArrays_returnEmpty` — loop skipped, no divide-by-zero
- `diffSquared_derivative_knownResult` — `2(x-y)/n`
- `diffSquared_derivative_perfectPredictionIsZero`

**crossEntropy** (5)
- `crossEntropy_calculate_knownResults`
- `crossEntropy_calculate_zeroProbabilityForTrueLabel_throwsAssertionError`
- `crossEntropy_calculate_unitProbabilityForFalseLabel_throwsAssertionError`
- `crossEntropy_derivative_trueLabel_isCorrect` — y=1, x=0.5 -> -2
- `crossEntropy_derivative_falseLabel_shouldMatchAnalyticGradient` — now passes after Cost.java:50 fix

**contract** (2)
- `calculate_withNaNOutput_throwsAssertionError`
- `derivative_withInfiniteOutput_throwsAssertionError`

### Planned Pinpoint Additions (20)

**diffSquared** (7)
- `diffSquared_calculate_swapInputAndExpected_sameCost` — `(x-y)^2 == (y-x)^2`
- `diffSquared_calculate_largePredictionError_scalesAsSquare`
- `diffSquared_calculate_negativeValues_areHandledLikePositives`
- `diffSquared_calculate_returnsNewArray_notAlias`
- `diffSquared_derivative_singleElement_knownResult`
- `diffSquared_derivative_emptyArrays_returnEmpty`
- `diffSquared_derivative_signFollowsErrorDirection` — overprediction -> positive grad

**crossEntropy** (8)
- `crossEntropy_calculate_falseLabelPerfectPrediction_isZero` — x=0, y=0 -> 0
- `crossEntropy_calculate_trueLabelPerfectPrediction_isZero` — x=1, y=1 -> 0
- `crossEntropy_calculate_partialProbability_matchesFormula` — y=0.5 (non-binary label) - currently asserts the correct mixed formula `-(y log x + (1-y) log(1-x))`; the forward branches on `y==1` and gets non-binary cases wrong (a 9th source bug — fail-loudly)
- `crossEntropy_calculate_emptyArrays_returnEmpty`
- `crossEntropy_calculate_returnsNewArray_notAlias`
- `crossEntropy_derivative_falseLabelZeroDivisor_throwsAssertionError` — x=1, y=0 -> 1/0
- `crossEntropy_derivative_singleElement_knownResult`
- `crossEntropy_derivative_emptyArrays_returnEmpty`

**contract / general** (5)
- `calculate_doesNotMutateOutputArgument`
- `derivative_doesNotMutateExpectedOutputArgument`
- `derivative_withNaNExpectedOutput_propagatesSilently` — no guard on `expectedOutput`
- `calculate_inputLengthLessThanExpected_throwsArrayIndexOutOfBounds`
- `calculate_inputLengthGreaterThanExpected_throwsArrayIndexOutOfBounds`

### Planned Class-Wise Black-Box (3)

- `diffSquared_derivative_matchesNumericalGradient` — central-difference vs analytic for random inputs
- `crossEntropy_derivative_matchesNumericalGradient` — same, over (0,1) range
- `bothCosts_areNonNegative_forValidInputs` — sweep 100 random valid input/expected pairs

---

## LayerTest

Abstract `Layer` exercised through an in-file `StubLayer`.

### Existing Tests (16)

**constructor** (1)
- `constructor_allocatesZeroedBiasAndGradient_velocitiesNull`

**initialize per optimizer** (4)
- `initialize_sgd_allocatesNoVelocityArrays`
- `initialize_sgdMomentum_allocatesVelocityOnly`
- `initialize_rmsProp_allocatesVelocitySquaredOnly`
- `initialize_adam_allocatesBothVelocityArrays`

**applyGradient math** (6)
- `applyGradient_sgd_subtractsScaledGradient`
- `applyGradient_sgdMomentum_updatesVelocityThenBias`
- `applyGradient_rmsProp_updatesVelocitySquaredThenBias`
- `applyGradient_adam_accumulatesVelocities`
- `applyGradient_adam_appliesBiasCorrectedUpdate` — now passes after Layer.java:93-94 fix
- `applyGradient_incrementsTimestepOnlyForAdam`

**getNumParameters / equals** (5)
- `getNumParameters_isBiasLength`
- `equals_identicalLayersAreEqual`
- `equals_differentNodeCountsAreNotEqual`
- `equals_differentBiasOrGradientAreNotEqual`
- `equals_nullVersusAllocatedVelocityAreNotEqual`

### Planned Pinpoint Additions (16)

**constructor / initialize** (5)
- `constructor_zeroNodes_allocatesEmptyArrays`
- `constructor_largeNodes_allocatesCorrectLength`
- `initialize_supplierCalledOncePerBias` — counter supplier, count == nodes
- `initialize_withNaNSupplier_makesBiasNaN` — no finite-guard at init
- `initialize_overridesExistingBias` — initialize twice, second wins

**applyGradient edge cases** (7)
- `applyGradient_zeroGradient_leavesBiasUnchanged_sgd`
- `applyGradient_zeroGradient_leavesBiasUnchanged_sgdMomentum`
- `applyGradient_zeroGradient_leavesBiasUnchanged_rmsProp`
- `applyGradient_zeroGradient_leavesBiasUnchanged_adam`
- `applyGradient_sgd_negativeLearningRate_biasGoesOpposite`
- `applyGradient_rmsProp_epsilonGuardsZeroVelocity` — eps=0 with zero velocitySquared -> no NaN
- `applyGradient_adam_largeT_correctionFactorApproachesOne`

**getNumParameters / equals / helper** (4)
- `getNumParameters_zeroNodes_isZero`
- `equals_differingTimestepStillEqual` — `t` is not part of `equals`
- `equals_acrossSubclassesWithMatchingFields` — Layer.equals doesn't check exact subclass
- `arraysDeepToString_formatsToTwoDecimalPlaces`

### Planned Class-Wise Black-Box (3)

- `optimizerConvergence_quadraticLoss` — many `applyGradient` calls with `biasGradient = bias - target` -> bias converges to target for each optimizer
- `adam_multiStepDecaysCorrectionTowardOne` — observe `t`, `correctionMomentum`, behaviour over 100 steps
- `roundTrip_initializeApplyClearEquals` — full sequence on two StubLayers leaves them equal

---

## DenseLayerTest

`DenseLayer` (package-private). `weights` is `private`; tests drive it through
`updateGradient`/`applyGradient` and observe via `calculateWeightedOutput`.

### Existing Tests (14)

**constructor / getNumParameters** (1)
- `getNumParameters_isWeightsPlusBiases` — `nodesBefore * nodes + nodes`

**calculateWeightedOutput** (2)
- `calculateWeightedOutput_freshLayerProducesZeroVectorOfOutputSize`
- `calculateWeightedOutput_wrongInputLength_throwsAssertionError`

**updateGradient / clearGradient** (3)
- `updateGradient_returnsArrayOfNodesBeforeLength`
- `updateGradient_accumulatesIntoBiasGradient`
- `clearGradient_resetsBiasGradient`

**applyGradient** (4)
- `applyGradient_sgd_updatesWeightsAndBiasObservably`
- `applyGradient_accumulatedGradientDoublesWeightEffect`
- `clearGradient_makesSubsequentApplyGradientANoOp`
- `initialize_thenApplyGradient_worksForEveryOptimizer`

**equals / clone / toString** (4)
- `equals_layerEqualsItself` — now passes after the `!super.equals` fix
- `clone_squareLayer_behavesIdenticallyToOriginal`
- `clone_nonSquareLayer_behavesIdenticallyToOriginal` — now passes after the `weights[0].length` fix
- `toString_mentionsBiases`

### Planned Pinpoint Additions (16)

**constructor / shape** (3)
- `constructor_oneByOne_works`
- `constructor_zeroNodesBefore_yieldsEmptyWeightsMatrix`
- `getNumParameters_variantShapes` — (5,1), (1,5), (10,10)

**calculateWeightedOutput** (3)
- `calculateWeightedOutput_returnsNewArray_notAlias`
- `calculateWeightedOutput_multiElementInputAndOutput_handCheckedAfterSgd` — known 2->3 result
- `calculateWeightedOutput_returnsArrayOfLengthNodes`

**updateGradient** (4)
- `updateGradient_zeroDzDc_leavesGradientsUnchanged`
- `updateGradient_nanDzDc_propagatesSilently` — Linalg.addInPlace has no finite-guard
- `updateGradient_returnsNewArray_notAlias`
- `updateGradient_xShorterThanNodesBefore_throwsArrayIndexOutOfBounds`

**applyGradient / clearGradient** (3)
- `applyGradient_adam_weightsAreBiasCorrected` — observable via two-step output drift
- `applyGradient_sgdMomentum_observableWeightChangeOverMultipleSteps`
- `clearGradient_calledTwice_isStillANoOp`

**equals / clone / toString** (3)
- `equals_sameShapeDifferentWeights_areNotEqual`
- `clone_independence_modifyingCloneDoesNotAffectOriginal` — applyGradient to clone, original unchanged
- `toString_containsWeightsAndBiases`

### Planned Class-Wise Black-Box (4)

- `learnsLinearMapping_sgd` — 1x1 layer trained with SGD converges on `y = 2x + 1`
- `learnsLinearMapping_eachOptimizer` — same problem, each of the 4 optimizers reaches the target within tolerance
- `numericalGradientMatchesUpdateGradient` — central-difference loss derivative ~ accumulated `weightsGradient` after a single `updateGradient`
- `multiStepTrainingReducesOutputError` — fixed input -> target, error decreases monotonically over K steps

---

## ConvolutionalLayerTest

`ConvolutionalLayer` (package-private). To keep convolution arithmetic hand-checkable despite
the internal reflection-padding index map, the smallest tests use a 2x2 input with a single 2x2
kernel and uniform inputs (so every kernel weight multiplies 1.0).

### Existing Tests (14)

**constructor / output-dim math** (4)
- `nodes_noPadding_followCeilDivFormula`
- `nodes_withPadding_preserveInputArea`
- `constructor_kernelWiderThanInput_throwsNegativeArraySize`
- `getNumParameters_isKernelsPlusBiases`

**calculateWeightedOutput** (2)
- `calculateWeightedOutput_freshLayerProducesZeroVectorOfNodeCount`
- `calculateWeightedOutput_wrongInputLength_throwsAssertionError`

**updateGradient / clearGradient** (2)
- `updateGradient_returnsDaDcSizedToTheInputVolume`
- `clearGradient_makesSubsequentApplyGradientANoOp`

**applyGradient** (2)
- `applyGradient_sgd_modifiesKernelsObservably`
- `applyGradient_rmsProp_keepsKernelsFinite` — now passes after the denominator fix

**equals / clone / toString** (4)
- `equals_layerEqualsItself` — now passes after the `!super.equals` fix
- `clone_squareKernel_behavesIdenticallyToOriginal`
- `clone_nonSquareKernel_behavesIdenticallyToOriginal` — now passes after the `kernelHeight` fix
- `toString_mentionsKernelsAndBiases`

### Planned Pinpoint Additions (18)

**constructor / shape** (4)
- `constructor_oneByOneKernel_works`
- `constructor_strideGreaterThanKernel_works`
- `constructor_multipleKernels_kernelsArrayShape` — verified via getNumParameters
- `getNumParameters_variantConfigs`

**calculateWeightedOutput** (4)
- `calculateWeightedOutput_singleKernelKnownResult` — hand-set kernel, hand-checked output
- `calculateWeightedOutput_multipleKernels_outputsStackedCorrectly` — output blocks per-kernel
- `calculateWeightedOutput_paddingTrue_doesNotThrow` — smoke; padding path is index-fragile
- `calculateWeightedOutput_returnsNewArray_notAlias`

**updateGradient** (4)
- `updateGradient_zeroKernels_daDcIsZero`
- `updateGradient_zeroDzDc_kernelsGradientUnchanged`
- `updateGradient_nanDzDc_throwsAssertionError` — explicit `assert Double.isFinite(dz_dC[...])`
- `updateGradient_returnsNewArray_notAlias`

**applyGradient / clearGradient** (4)
- `applyGradient_sgdMomentum_modifiesKernelsObservably`
- `applyGradient_adam_modifiesKernelsObservably`
- `applyGradient_negativeGradient_kernelsGoPositive`
- `clearGradient_calledTwice_isStillANoOp`

**equals / clone** (2)
- `equals_sameConfigDifferentKernels_areNotEqual` — drive different kernel state, equals false
- `clone_independence_modifyingCloneDoesNotAffectOriginal`

### Planned Class-Wise Black-Box (4)

- `learnsConstantKernel_sgd` — input fixed at ones, target output fixed -> kernel converges to constant value
- `learnsConstantKernel_eachOptimizer` — same with all 4 optimizers
- `numericalGradientMatchesUpdateGradient` — central-difference vs accumulated `kernelsGradient`
- `multiKernelSuperposition` — output of (2 kernels) == column-stack of running each kernel alone

---

## NNTest

`NN` plus its `NetworkBuilder`.

### Existing Tests (16)

**NetworkBuilder validation** (6)
- `build_missingInputNum_throwsMissingInformation`
- `build_missingHiddenAF_throwsMissingInformation`
- `build_missingOutputAF_throwsMissingInformation`
- `build_missingCostFunction_throwsMissingInformation`
- `build_noLayers_throwsMissingInformation`
- `build_fullySpecified_producesNonNullNetwork`

**NetworkBuilder ordering / dimensions** (3)
- `setInputNum_afterAddingLayer_throwsUnsupportedOperation`
- `addDenseLayer_beforeSetInputNum_throwsNegativeArraySize`
- `addConvolutionalLayer_inputDimsMismatch_throwsAssertionError`

**calculateOutput** (2)
- `calculateOutput_wrongInputLength_throwsAssertionError`
- `calculateOutput_softmaxWithZeroTemperature_throwsAssertionError`

**learn / learnSingleOutput** (3)
- `learn_emptyBatch_throwsAssertionError`
- `learn_mismatchedBatchLengths_throwsAssertionError`
- `learnSingleOutput_outputIndexOutOfRange_throwsAssertionError`

**equals / clone** (2)
- `equals_networkEqualsItself`
- `clone_producesEqualNetwork` — now passes after the DenseLayer.clone fix

### Planned Pinpoint Additions (20)

**NetworkBuilder coverage** (5)
- `build_eachOptimizer_succeeds` — SGD, MOMENTUM, RMS_PROP, ADAM
- `build_eachHiddenActivation_succeeds`
- `build_eachOutputActivation_succeeds`
- `build_eachCostFunction_succeeds`
- `addCustomLayer_extendsLayerStack`

**calculateOutput** (5)
- `calculateOutput_returnsArrayOfLengthOutputNum`
- `calculateOutput_softmaxOutputSumsToOne` — with non-zero temperature
- `calculateOutput_isDeterministic` — same input twice -> identical output
- `calculateOutput_temperatureFlatensSoftmax` — large T -> closer to uniform
- `calculateOutput_temperatureIgnored_forNonSoftmaxOutputs`

**setTemperature** (1)
- `setTemperature_defaultIsOne`

**calculateCost** (3)
- `calculateCost_perfectPrediction_isZero_diffSquared`
- `calculateCost_nonNegative_forValidInputs`
- `calculateCost_returnsScalarSumOfPerElementCosts`

**backPropagate / learn** (4)
- `backPropagate_validInput_doesNotThrow`
- `backPropagate_populatesLayerGradients` — observable via subsequent applyGradient drift
- `learn_singleStep_changesOutput`
- `learnSingleOutput_validIndex_doesNotThrow`

**clone / toString** (2)
- `clone_trainingClone_doesNotAffectOriginal`
- `toString_containsParameterCountAndLayerSections`

### Planned Class-Wise Black-Box (5)

- `trainsAndLogicGate` — 2->4->2 + ADAM + crossEntropy + softmax, K steps; final accuracy on 4 cases >= 75%
- `trainsTinyLinearRegression` — 1->1 + SGD + diffSquared, K steps -> mean squared error within tolerance
- `allOptimizers_canTrainWithoutNaN` — same XOR-style problem with each optimizer, K steps, weights stay finite
- `convolutionalNetwork_smokeTrain` — tiny conv + dense network, K steps, no exceptions and output stays finite
- `cloneRoundTrip_originalUnaffectedByCloneTraining` — clone, train clone for K steps, original output unchanged

---

## Notes on Implementation

- **Determinism.** Where a test depends on weight values, use freshly-constructed pre-`initialize`
  layers (all-zero) or use `initialize(() -> 0.0, optimizer)`. For black-box training tests,
  reseed the activation `Random` if exposed or accept stochastic variation with a generous tolerance.
- **Aliasing assertions.** "returns new array" tests should compare references with `assertNotSame`
  and verify the input is unchanged by snapshotting then equality-checking after the call.
- **Numerical-gradient checks.** Central differences with `h = 1e-5`, tolerance `1e-3` on the
  L-infinity error. Compute analytic gradient via the function under test, compare component-wise.
- **Black-box training tests.** Use small step counts (50-500) and check direction-of-change
  (loss decreased, accuracy increased) rather than absolute values to keep the suite fast and
  resistant to RNG variation.
- **Run with** `mvn clean test` — see the memory note on the IDE-vs-Maven JDK mismatch.
