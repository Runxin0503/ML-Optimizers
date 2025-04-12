package Network;

/**
 * Enum representing various optimizers used in training neural networks.
 * <p>
 * Each optimizer determines the update rule used during training to adjust the weights and biases of the model.
 * These optimizers are crucial for improving the convergence of the training process by reducing the loss function.
 * <ul>
 *     <li><strong>SGD (Stochastic Gradient Descent)</strong>: Basic gradient descent method that updates weights based on the gradient of the loss function.</li>
 *     <li><strong>SGD with Momentum</strong>: Enhances SGD by adding momentum, which helps accelerate the convergence and reduces oscillations.</li>
 *     <li><strong>RMSProp</strong>: Optimizer that adjusts the learning rate based on the recent magnitudes of the gradients.</li>
 *     <li><strong>Adam (Adaptive Moment Estimation)</strong>: Combines the advantages of both RMSProp and momentum, adapting the learning rate based on both the first and second moments of the gradients.</li>
 * </ul>
 * The enum allows for the identification of the optimizer type being used, but the actual implementation of the update rule would typically be handled within the corresponding optimizer class.
 */
public enum Optimizer {
    /**
     * Stochastic Gradient Descent (SGD): Updates the parameters using the gradient of the loss function with respect to the weights.
     * <p>
     * Formula:
     * <pre>
     * {@code w = w - learning_rate * gradient}
     * </pre>
     * Where:
     * <ul>
     *     <li>{@code w} is the weight</li>
     *     <li>{@code learning_rate} is the step size for weight updates</li>
     *     <li>{@code gradient} is the partial derivative of the loss function with respect to the weight</li>
     * </ul>
     */
    SGD,

    /**
     * Stochastic Gradient Descent with Momentum: Adds momentum to the gradient update to accelerate convergence and reduce oscillations.
     * <p>
     * Formula:
     * <pre>
     * {@code v = momentum * v + learning_rate * gradient
     * w = w - v}
     * </pre>
     * Where:
     * <ul>
     *     <li>{@code v} is the velocity (a running average of gradients)</li>
     *     <li>{@code momentum} is a hyperparameter that determines the weight of past gradients</li>
     *     <li>{@code learning_rate} is the step size for weight updates</li>
     *     <li>{@code gradient} is the partial derivative of the loss function with respect to the weight</li>
     * </ul>
     */
    SGD_MOMENTUM,

    /**
     * RMSProp (Root Mean Square Propagation): Divides the learning rate by a running average of recent gradient magnitudes to adjust for varying gradient sizes.
     * <p>
     * Formula:
     * <pre>
     * {@code v = beta * v + (1 - beta) * gradient^2
     * w = w - learning_rate * gradient / (sqrt(v) + epsilon)}
     * </pre>
     * Where:
     * <ul>
     *     <li>{@code v} is the moving average of squared gradients</li>
     *     <li>{@code beta} is the hyperparameter that determines the weight of past squared gradients</li>
     *     <li>{@code learning_rate} is the step size for weight updates</li>
     *     <li>{@code gradient} is the partial derivative of the loss function with respect to the weight</li>
     *     <li>{@code epsilon} is a small constant added to prevent division by zero</li>
     * </ul>
     */
    RMS_PROP,

    /**
     * Adam (Adaptive Moment Estimation): Combines the benefits of both momentum and RMSProp. Uses both the first moment (mean) and the second moment (uncentered variance) of gradients to adapt the learning rate.
     * <p>
     * Formula:
     * <pre>
     * {@code m = beta1 * m + (1 - beta1) * gradient
     * v = beta2 * v + (1 - beta2) * gradient^2
     * m_hat = m / (1 - beta1^t)
     * v_hat = v / (1 - beta2^t)
     * w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)}
     * </pre>
     * Where:
     * <ul>
     *     <li>{@code m} is the first moment (mean of gradients)</li>
     *     <li>{@code v} is the second moment (uncentered variance of gradients)</li>
     *     <li>{@code momentum} is a hyperparameter that determines the weight of past gradients</li>
     *     <li>{@code beta} is the hyperparameter that determines the weight of past squared gradients</li>
     *     <li>{@code learning_rate} is the step size for weight updates</li>
     *     <li>{@code gradient} is the partial derivative of the loss function with respect to the weight</li>
     *     <li>{@code epsilon} is a small constant added to prevent division by zero</li>
     *     <li>{@code m_hat} and {@code v_hat} are bias-corrected estimates of the first and second moments</li>
     * </ul>
     */
    ADAM
}
