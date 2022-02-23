package org.deeplearning4j.rl4j.examples.advanced.src.main.java;

import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

class State implements Encodable {
    // each element in the list is a state-action representation
    // number of entries in the list is equal to number of available actions
    private final ArrayList<double[]> inputs;

    public State(ArrayList<double[]> inputs) {
        this.inputs = inputs;
    }

    public double[] toArray() {
        return null; // deprecated
    }

    public boolean isSkipped() {
        return false;
    }

    public int getNumberOfAvailableActions() {
        return inputs.size();
    }

    public INDArray getData() {
        int availActions = getNumberOfAvailableActions();
        double[][][] inputsStacked = new double[1][availActions][];
        for (int action = 0; action < availActions; action++) {
            inputsStacked[0][action] = inputs.get(action);
        }
        return Nd4j.create(inputsStacked);
    }

    public Encodable dup() {
        return new State(inputs);
    }
}
