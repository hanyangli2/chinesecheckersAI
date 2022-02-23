package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;


public class DQNVarActions<NN extends org.deeplearning4j.rl4j.network.dqn.DQN> extends DQN<NN>
{
    int[] numInputs;

    public DQNVarActions(MultiLayerNetwork mln, int[] numInputs)
    {
        super(mln);
        this.numInputs = new int[numInputs.length];
        System.arraycopy(numInputs, 0, this.numInputs, 0, numInputs.length);
    }

    public NN clone() {
        NN nn = (NN)new DQNVarActions<>(mln.clone(), numInputs);
        nn.mln.setListeners(mln.getListeners());
        return nn;
    }

    public DQNVarActions(MultiLayerNetwork mln) {
        super(mln);
    }

    public static DQNVarActions load(String path) throws IOException
    {
        return new DQNVarActions(ModelSerializer.restoreMultiLayerNetwork(path));
    }

    public void fit(INDArray input, INDArray labels) {
        long[] shape = input.shape();
        long[] newShape = new long[shape.length - 1];
        System.arraycopy(shape, 1, newShape, 0, newShape.length);
        INDArray reshapedInput = input.reshape(newShape);
        INDArray reshapedLabels = labels.reshape(labels.length(), 1);
        mln.fit(reshapedInput, reshapedLabels);
    }

    public INDArray output(INDArray batch)
    {
        long[] shape = batch.shape();
        long[] newShape = new long[shape.length - 1];
        System.arraycopy(shape, 1, newShape, 0, newShape.length);
        INDArray reshapedBatch = batch.reshape(newShape);
        INDArray out = mln.output(reshapedBatch);
        INDArray outReshaped = out.reshape(1, out.length());
        return outReshaped;
    }

    public INDArray output(Observation observation)
    {
        return this.output(observation.getData());
    }


    public double getLatestScore() {
        return mln.score();
    }
}
