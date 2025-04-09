package Network;

import Utils.Linalg;
import enums.Activation;
import enums.Optimizer;

import java.util.Arrays;
import java.util.function.Supplier;

/** A type of RNN Layer that remembers both short-term and long-term data from past training */
public class LSTMLayer extends Layer {

    /** The input length of this Layer */
    private final int nodesBefore;

    /** The short-term memory storage (and output) for this Recurrent Neural Network (RNN) Layer. */
    private double[] hPrev;

    /** The long-term (Candidate) memory storage for this Recurrent Neural Network (RNN) Layer. */
    private double[] CPrev;

    /** The Forget Gate (first gate) layer storing weights for the previous output (h). */
    private DenseLayer fGate_h;

    /** The Forget Gate (first gate) layer storing weights for the current input (x). */
    private DenseLayer fGate_x;

    /** The weights for the new Candidate values (second gate) layer, storing weights for the previous output (h). */
    private DenseLayer iGate_h;

    /** The weights for the new Candidate values (second gate) layer, storing weights for the current input (x). */
    private DenseLayer iGate_x;

    /** The constructor for the new Candidate values (second gate) layer,
     * storing weights for the previous output (h). */
    private DenseLayer cGate_h;

    /** The constructor for the new Candidate values (second gate) layer,
     * storing weights for the current input (x). */
    private DenseLayer cGate_x;

    /** The weights of the new Candidate values (second gate) layer, storing weights for the previous output (h). */
    private DenseLayer oGate_h;

    /** The weights of the new Candidate values (second gate) layer, storing weights for the current input (x). */
    private DenseLayer oGate_x;

    public LSTMLayer(int prevNodes, int nodes) {
        super(nodes);
        nodesBefore = prevNodes;
        hPrev = new double[nodes];
        CPrev = new double[nodes];
        fGate_h = new DenseLayer(nodes, nodes);
        fGate_x = new DenseLayer(prevNodes, nodes);
        iGate_h = new DenseLayer(nodes, nodes);
        iGate_x = new DenseLayer(prevNodes, nodes);
        cGate_h = new DenseLayer(nodes, nodes);
        cGate_x = new DenseLayer(prevNodes, nodes);
        oGate_h = new DenseLayer(nodes, nodes);
        oGate_x = new DenseLayer(prevNodes, nodes);
    }

    @Override
    public double[] calculateWeightedOutput(double[] input) {
        return (
                hPrev = Linalg.multiply(
                        Activation.sigmoid.calculate(
                                Linalg.add(
                                        oGate_h.calculateWeightedOutput(hPrev),
                                        oGate_x.calculateWeightedOutput(input)
                                )
                        ),
                        Activation.tanh.calculate(
                                (CPrev = Linalg.add(
                                        Linalg.multiply(
                                                CPrev,
                                                Activation.sigmoid.calculate(
                                                        Linalg.add(
                                                                fGate_h.calculateWeightedOutput(hPrev),
                                                                fGate_x.calculateWeightedOutput(input)
                                                        )
                                                )
                                        ),
                                        Linalg.multiply(
                                                Activation.sigmoid.calculate(
                                                        Linalg.add(
                                                                iGate_h.calculateWeightedOutput(hPrev),
                                                                iGate_x.calculateWeightedOutput(input)
                                                        )
                                                ),
                                                Activation.tanh.calculate(
                                                        Linalg.add(
                                                                cGate_h.calculateWeightedOutput(hPrev),
                                                                cGate_x.calculateWeightedOutput(input)
                                                        )
                                                )
                                        )
                                )
                                )
                        )
                )
        );
    }

    @Override
    void initialize(Supplier<Double> initializer, Optimizer optimizer) {
        fGate_h.initialize(initializer, optimizer);
        fGate_x.initialize(initializer, optimizer);
        iGate_h.initialize(initializer, optimizer);
        iGate_x.initialize(initializer, optimizer);
        cGate_h.initialize(initializer, optimizer);
        cGate_x.initialize(initializer, optimizer);
        oGate_h.initialize(initializer, optimizer);
        oGate_x.initialize(initializer, optimizer);
    }

    @Override
    public int getNumParameters() {
        return fGate_h.getNumParameters() + fGate_x.getNumParameters() +
                iGate_h.getNumParameters() + iGate_x.getNumParameters() +
                cGate_h.getNumParameters() + cGate_x.getNumParameters() +
                oGate_h.getNumParameters() + oGate_x.getNumParameters();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof LSTMLayer o)) return false;
        return fGate_h.equals(o.fGate_h) && fGate_x.equals(o.fGate_x) &&
                iGate_h.equals(o.iGate_h) && iGate_x.equals(o.iGate_x) &&
                cGate_h.equals(o.cGate_h) && cGate_x.equals(o.cGate_x) &&
                oGate_h.equals(o.oGate_h) && oGate_x.equals(o.oGate_x);
    }

    /** Returns an array of {@code Previous Output} and {@code Previous Candidate Data} used in RNNs. */
    public double[][] getPrevMemories() {
        return new double[][]{hPrev, CPrev};
    }


    /** Returns an array of {@code Previous Output} and {@code Previous Candidate Data} used in RNNs. */
    public void clearPrevMemories() {
        Arrays.fill(hPrev,0);
        Arrays.fill(CPrev,0);
    }

    @Override
    double[] updateGradient(double[] dz_dC, double[] x) {
        throw new UnsupportedOperationException("Not supported for RNN layers.");
    }

    /**
     * Given the derivative array of this layer's output w.r.t the loss function (dz_dC),
     * the previous input of this layer,
     * and the previous memories of this RNN layer,
     * calculate and shift this layer's gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    double[] updateGradient(double[] dz_dC, double[] x, double[] hPrev, double[] CPrev) {
        double[] forgetGateH = fGate_h.calculateWeightedOutput(hPrev),
                forgetGateX = fGate_x.calculateWeightedOutput(x);
        double[] forget = Linalg.add(
                forgetGateH,
                forgetGateX
        );
        double[] forgetNormalized = Activation.sigmoid.calculate(
                forget
        );

        double[] CWeightGateH = iGate_h.calculateWeightedOutput(hPrev),
                CWeightGateX = iGate_x.calculateWeightedOutput(x);
        double[] newCandidateWeight = Linalg.add(
                CWeightGateH,
                CWeightGateX
        );
        double[] newCandidateWeightNormalized = Activation.sigmoid.calculate(
                newCandidateWeight
        );

        double[] CandidateGateH = cGate_h.calculateWeightedOutput(hPrev),
                CandidateGateX = cGate_x.calculateWeightedOutput(x);
        double[] newCandidateConstructor = Linalg.add(
                CandidateGateH,
                CandidateGateX
        );
        double[] newCandidateConstructorNormalized = Activation.tanh.calculate(
                newCandidateConstructor
        );
        double[] newCandidate = Linalg.multiply(
                newCandidateWeightNormalized,
                newCandidateConstructorNormalized
        );

        double[] CPrevForget = Linalg.multiply(
                CPrev,
                forgetNormalized
        );

        double[] CNew = Linalg.add(
                CPrevForget,
                newCandidate
        );

        double[] CNewTanh = Activation.tanh.calculate(
                CNew
        );

        double[] outputWeightH = oGate_h.calculateWeightedOutput(hPrev),
                outputWeightX = oGate_x.calculateWeightedOutput(x);
        double[] outputWeight = Linalg.add(
                outputWeightH,
                outputWeightX
        );
        double[] outputWeightNormalized = Activation.sigmoid.calculate(
                outputWeight
        );

        double[] oGateSigmoidDeriv = Activation.sigmoid.derivative(
                outputWeight,
                Linalg.multiply(
                        dz_dC,
                        CNewTanh
                )
        );
        oGate_h.updateGradient(
                oGateSigmoidDeriv,
                hPrev
        );
        double[] outputGateDeriv = oGate_x.updateGradient(
                oGateSigmoidDeriv,
                x
        );

        double[] CandidateTanhDeriv = Activation.tanh.derivative(CNew, Linalg.multiply(outputWeightNormalized, dz_dC));
        double[] cGateTanhDeriv = Activation.tanh.derivative(
                Linalg.multiply(
                        newCandidateWeightNormalized,
                        CandidateTanhDeriv
                ),
                newCandidateConstructor
        );

        cGate_h.updateGradient(
                cGateTanhDeriv,
                hPrev
        );
        double[] cGateDeriv = cGate_x.updateGradient(
                cGateTanhDeriv,
                x
        );

        double[] cWeightSigmoidDeriv = Activation.sigmoid.derivative(
                Linalg.multiply(
                        newCandidateConstructorNormalized,
                        CandidateTanhDeriv
                ),
                newCandidateWeight
        );
        iGate_h.updateGradient(
                cWeightSigmoidDeriv,
                hPrev
        );
        double[] iGateDeriv = iGate_x.updateGradient(
                cWeightSigmoidDeriv,
                x
        );

        double[] forgetGateSigmoidDeriv = Activation.sigmoid.derivative(
                Linalg.multiply(
                        CandidateTanhDeriv,
                        CPrev
                ),
                forget
        );
        fGate_h.updateGradient(
                forgetGateSigmoidDeriv,
                hPrev
        );
        double[] fGateDeriv = fGate_x.updateGradient(
                forgetGateSigmoidDeriv,
                x
        );

        return Linalg.add(
                Linalg.add(
                        outputGateDeriv,
                        cGateDeriv
                ),
                Linalg.add(
                        iGateDeriv,
                        fGateDeriv
                )
        );
    }

    @Override
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        fGate_h.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        fGate_x.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        iGate_h.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        iGate_x.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        cGate_h.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        cGate_x.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        oGate_h.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
        oGate_x.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
    }

    @Override
    void clearGradient() {
        fGate_h.clearGradient();
        fGate_x.clearGradient();
        iGate_h.clearGradient();
        iGate_x.clearGradient();
        cGate_h.clearGradient();
        cGate_x.clearGradient();
        oGate_h.clearGradient();
        oGate_x.clearGradient();
    }

    @Override
    public String toString() {
        return "----------Forget Gate (Previous Output)----------\n" + fGate_h +
                "\n----------Forget Gate (Current Input)----------\n" + fGate_x +
                "\n----------Candidate Gate Weights (Previous Output)----------\n" + iGate_h +
                "\n----------Candidate Gate Weights (Current Input)----------\n" + iGate_x +
                "\n----------Candidate Gate Constructor (Previous Output)----------\n" + cGate_h +
                "\n----------Candidate Gate Constructor (Current Input)----------\n" + cGate_x +
                "\n----------Output Gate Weights (Previous Output)----------\n" + oGate_h +
                "\n----------Output Gate Weights (Current Input)----------\n" + oGate_x;
    }

    @Override
    public Object clone() {
        LSTMLayer newLayer = new LSTMLayer(nodesBefore, fGate_x.nodes);
        newLayer.hPrev = hPrev.clone();
        newLayer.CPrev = CPrev.clone();
        newLayer.fGate_h = (DenseLayer) fGate_h.clone();
        newLayer.fGate_x = (DenseLayer) fGate_x.clone();
        newLayer.iGate_h = (DenseLayer) iGate_h.clone();
        newLayer.iGate_x = (DenseLayer) iGate_x.clone();
        newLayer.cGate_h = (DenseLayer) cGate_h.clone();
        newLayer.cGate_x = (DenseLayer) cGate_x.clone();
        newLayer.oGate_h = (DenseLayer) oGate_h.clone();
        newLayer.oGate_x = (DenseLayer) oGate_x.clone();
        return newLayer;
    }
}
