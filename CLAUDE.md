# ML-Optimizers

A from-scratch Java neural-network library focused on comparing gradient-based optimizers (SGD, SGD+Momentum, RMSProp, Adam). Single Maven artifact `io.github.runxin0503:ml-optimizers:1.0.0`, no runtime dependencies, JUnit 5 only for tests.

## Build & test

- Maven project. Source/target: Java 21 (see [pom.xml](pom.xml)). Surefire is pinned to 3.2.5 so JUnit 5 actually runs.
- Always run `mvn clean test` — the IDE may compile with Java 25, but Maven runs Java 21; a plain `mvn test` against IDE-built classes throws `UnsupportedClassVersionError`.
- Assertions are enabled in tests (`<enableAssertions>true</enableAssertions>`). The codebase uses `assert` heavily as a real contract layer; do not assume asserts are no-ops.
- Tests under [src/test/java/Network/](src/test/java/Network/) are *intentionally red* — they encode confirmed bugs in `Network/*`. Failures there are expected and should not be "fixed" by weakening the test.

## Package layout

Everything lives in one package: `Network` ([src/main/java/Network/](src/main/java/Network/)). No subpackages.

| File | Role |
|---|---|
| [NN.java](src/main/java/Network/NN.java) | Top-level network. Holds layers + global activations/cost/optimizer. Inner `NetworkBuilder` is the only public construction path. Inner `MissingInformationException` is thrown by `build()` when required fields are unset. |
| [Layer.java](src/main/java/Network/Layer.java) | Abstract base. Owns bias, biasGradient, optional biasVelocity / biasVelocitySquared (lazy-allocated per optimizer), and the per-layer Adam time step `t`. Implements the bias update for all four optimizers; subclasses override `applyGradient` and call `super.applyGradient` for biases. |
| [DenseLayer.java](src/main/java/Network/DenseLayer.java) | Package-private fully-connected layer. Weight matrix shape: `[nodesBefore][nodes]`. |
| [ConvolutionalLayer.java](src/main/java/Network/ConvolutionalLayer.java) | Package-private 2D conv layer with stride and optional padding. Kernels are `double[numKernels][kernelHeight][kernelWidth]`; output is flattened. Input/output are flat `double[]` of size `width*height*depth`. |
| [Activation.java](src/main/java/Network/Activation.java) | Enum: `none, ReLU, sigmoid, tanh, LeakyReLU, softmax`. Each constant carries both forward and `f'(z) * da/dC` backward. `getInitializer` picks He for ReLU/LeakyReLU, Xavier otherwise. |
| [Cost.java](src/main/java/Network/Cost.java) | Enum: `diffSquared` (MSE) and `crossEntropy`. Both forward and derivative. |
| [Optimizer.java](src/main/java/Network/Optimizer.java) | Enum tag only: `SGD, SGD_MOMENTUM, RMS_PROP, ADAM`. Update rules live in `Layer`/`DenseLayer`/`ConvolutionalLayer`. |
| [Linalg.java](src/main/java/Network/Linalg.java) | Tiny array helper: `matrixMultiply`, `dotProduct`, `multiply` (Hadamard), `scale`/`scaleInPlace`, `add`/`addInPlace`, `sum`. Most operations use `IntStream.parallel()`. |

## How the pieces fit

- Build a network with `new NN.NetworkBuilder().setInputNum(...).addDenseLayer(...).addConvolutionalLayer(...).setHiddenAF(...).setOutputAF(...).setCostFunction(...).setOptimizer(...).build()`. `setInputNum` must come before any `add*Layer` call. `build()` initializes each layer's weights using the AF-appropriate initializer.
- Inference: `nn.calculateOutput(double[] input)` runs forward through layers; hidden activation is applied between layers; output activation is applied last. If output AF is `softmax`, logits are divided by `temperature` before softmax (RL exploration).
- Training entry points are static on `NN`:
  - `NN.learn(nn, lr, momentum, beta, epsilon, inputs[][], outputs[][])` — spawns one `Thread` per training example to accumulate gradients in parallel, then applies a single averaged step (`lr / batchSize`). Synchronized on the NN instance.
  - `NN.learnSingleOutput(nn, lr, momentum, beta, epsilon, input, outputIndex, expectedOutput)` — RL-style single-scalar update: forward once, overwrite one output slot with the target, backprop.
- Backprop convention: `dz_dC` is gradient of pre-activation w.r.t. cost; `da_dC` is gradient of activation output w.r.t. cost. Layers expose `updateGradient(dz_dC, x)` returning `da_dC` for the previous layer.
- Optimizer state (`biasVelocity`, `biasVelocitySquared`, and the weight equivalents) is allocated only for the optimizers that need it — check for `null` before touching velocity arrays. Adam's bias-correction time step `t` is per-layer and incremented in the bias update.

## Things to watch when editing

- Required hyperparameters depend on optimizer (documented on `NN.learn`): SGD uses `lr` only; SGD_MOMENTUM uses `lr, momentum`; RMS_PROP uses `lr, beta, epsilon`; ADAM uses all four. Extra args are ignored, missing args silently no-op into the velocity update.
- `NN.backPropagate` divides the final-layer pre-activation by `temperature` *and* divides the softmax output by `temperature` again — that second division on `output` is a known oddity, not an intended scaling step.
- `Cost.crossEntropy.derivative` includes a hand-coded `(input[i] == 0 ? 0 : ...)` guard but has no symmetric guard for `input == 1`; numerical issues with sigmoid+CE are real.
- `Layer.equals` returns `false` when `super.equals` is true in `DenseLayer.equals` (`|| super.equals(obj)` should be `&&`) — known bug surfaced by the test suite.
- `DenseLayer.clearGradient` reallocates each row instead of zeroing in place; fine, but don't hold references to gradient rows across calls.
- Use `Linalg` helpers rather than hand loops — they're parallelized and the rest of the code expects new arrays vs. in-place semantics matching the `*InPlace` suffix.
- This is library code only; no `main`, no CLI, no examples module. Demos live in the test suite.
