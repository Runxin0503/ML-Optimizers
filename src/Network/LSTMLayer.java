package Network;

import Utils.Linalg;
import enums.Activation;
import enums.Optimizer;

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

    public LSTMLayer(int prevNodes,int nodes) {
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
        double[] forget = Activation.sigmoid.calculate(
                Linalg.add(
                        fGate_h.calculateWeightedOutput(hPrev),
                        fGate_x.calculateWeightedOutput(input)
                        )
        );
        double[] newCandidate = Linalg.multiply(
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
        );
        CPrev = Linalg.add(
                Linalg.multiply(
                    CPrev,
                    forget
                ),
                newCandidate
        );

        double[] output = Linalg.multiply(
                Activation.sigmoid.calculate(
                        Linalg.add(
                                oGate_h.calculateWeightedOutput(hPrev),
                                oGate_x.calculateWeightedOutput(input)
                        )
                ),
                Activation.tanh.calculate(
                        CPrev
                )
        );

        hPrev = output;
        return output;
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
        if(!(obj instanceof LSTMLayer o)) return false;
        return fGate_h.equals(o.fGate_h) && fGate_x.equals(o.fGate_x) &&
                iGate_h.equals(o.iGate_h) && iGate_x.equals(o.iGate_x) &&
                cGate_h.equals(o.cGate_h) && cGate_x.equals(o.cGate_x) &&
                oGate_h.equals(o.oGate_h) && oGate_x.equals(o.oGate_x);
    }

    @Override
    double[] updateGradient(double[] dz_dC, double[] x) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    void clearGradient() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String toString() {
        throw new UnsupportedOperationException("Not supported yet.");
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
