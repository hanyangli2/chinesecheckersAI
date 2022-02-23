package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm.DoubleDQN;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm.StandardDQN;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

public class QLearningDiscreteDenseVarActions<OBSERVATION extends Encodable> extends QLearningDiscreteDense<OBSERVATION>
{
    private static <A> INDArray buildStackedObservations(List<Transition<A>> transitions) {
        int size = transitions.size();
        INDArray[] array = new INDArray[size];

        INDArray observation = ((Transition)transitions.get(0)).getObservation().getData();
        long[] shape = observation.shape();
        long[] newShape = new long[shape.length];
        System.arraycopy(shape, 0, newShape, 0, newShape.length);
        long totalEntries = 0;

        for(int i = 0; i < size; ++i) {
            array[i] = ((Transition)transitions.get(i)).getObservation().getData();
            long[] sh = array[i].shape();
            long[] newSh = new long[sh.length - 1];
            System.arraycopy(sh, 1, newSh, 0, newSh.length);
            array[i] = array[i].reshape(newSh);
            totalEntries += array[i].shape()[0];
        }

        newShape[1] = totalEntries;
        return Nd4j.concat(0, array).reshape(newShape);
    }

    private static <A> INDArray buildStackedNextObservations(List<Transition<A>> transitions) {
        int size = transitions.size();
        INDArray[] array = new INDArray[size];

        INDArray nextObservation = transitions.get(0).getNextObservation();
        long[] shape = nextObservation.shape();
        long[] newShape = new long[shape.length];
        System.arraycopy(shape, 0, newShape, 0, newShape.length);
        long totalEntries = 0;

        for(int i = 0; i < size; ++i) {
            Transition<A> trans = (Transition)transitions.get(i);
            INDArray obs = trans.getObservation().getData();
            long historyLength = obs.shape()[0];
            if (historyLength != 1L)
            {
                throw new RuntimeException("Not Implemented!");
            }
            else
            {
                array[i] = trans.getNextObservation();
                long[] sh = array[i].shape();
                long[] newSh = new long[sh.length - 1];
                System.arraycopy(sh, 1, newSh, 0, newSh.length);
                array[i] = array[i].reshape(newSh);
                totalEntries += array[i].shape()[0];
            }
        }

        newShape[1] = totalEntries;
        return Nd4j.concat(0, array).reshape(newShape);
    }

    private static <A> long[] getShape(List<Transition<A>> transitions) {
        INDArray observations = ((Transition)transitions.get(0)).getObservation().getData();
        long[] observationShape = observations.shape();
        long[] stackedShape;
        long totalEntries = 0L;
        if (observationShape[0] == 1L)
        {
            stackedShape = new long[observationShape.length];
            System.arraycopy(observationShape, 0, stackedShape, 0, observationShape.length);
            for (Transition<A> transition : transitions)
            {
                INDArray obs = ((Transition) transition).getObservation().getData();
                long[] obsShape = obs.shape();
                totalEntries += obsShape[1];
            }
        }
        else
        {
            throw new RuntimeException("Not Implemented!");
        }

        stackedShape[1] = totalEntries;
        stackedShape[0] = 1;
        return stackedShape;
    }



    public QLearningDiscreteDenseVarActions(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, DQNFactory factory, QLearningConfiguration conf)
    {
        super(mdp, factory, conf);
        tdTargetAlgorithm = conf.isDoubleDQN() ?
            new DoubleDQN(this, conf.getGamma(), conf.getErrorClamp())
            {
                // In litterature, this corresponds to: max_{a}Q(s_{t+1}, a)
                private int[] maxActionsFromQNetworkNextObservation;

                protected void initComputation(INDArray observations, INDArray nextObservations, List<Transition<Integer>> transitions)
                {
                    super.initComputation(observations, nextObservations);
                    maxActionsFromQNetworkNextObservation = new int[transitions.size()];
                    INDArray qTmp = qNetworkNextObservation.reshape(qNetworkNextObservation.length());
                    long pos = 0;
                    for (int i = 0; i < transitions.size(); i++)
                    {
                        Transition<Integer> currentTransition = transitions.get(i);
                        long actionCount = currentTransition.getNextObservation().shape()[1];
                        maxActionsFromQNetworkNextObservation[i] = Nd4j.argMax(qTmp.get(NDArrayIndex.interval(pos, pos + actionCount)), 0).getInt(0);
                        pos += actionCount;
                    }
                }

                protected double computeTarget(int batchIdx, long posNext, double reward, boolean isTerminal) {
                    double yTarget = reward;
                    if (!isTerminal) {
                        yTarget += gamma * targetQNetworkNextObservation.getDouble(0, posNext + maxActionsFromQNetworkNextObservation[batchIdx]);
                    }

                    return yTarget;
                }

                public DataSet computeTDTargets(List<Transition<Integer>> transitions)
                {

                    double errorClamp = conf.getErrorClamp();
                    boolean isClamped = !Double.isNaN(errorClamp);

                    int size = transitions.size();

                    INDArray observations = buildStackedObservations(transitions);
                    INDArray nextObservations = buildStackedNextObservations(transitions);

                    initComputation(observations, nextObservations, transitions);

                    INDArray updatedQValues = qNetworkSource.getQNetwork().output(observations);
                    long posCurrent = 0;
                    long posNext = 0;
                    for (int i = 0; i < size; ++i)
                    {
                        Transition<Integer> transition = transitions.get(i);
                        long actionCountCurrent = transition.getObservation().getData().shape()[1];
                        long actionCountNext = transition.getNextObservation().shape()[1];

                        double yTarget = computeTarget(i, posNext, transition.getReward(), transition.isTerminal());

                        if(isClamped) {
                            double previousQValue = updatedQValues.getDouble(0, posCurrent + transition.getAction());
                            double lowBound = previousQValue - errorClamp;
                            double highBound = previousQValue + errorClamp;
                            yTarget = Math.min(highBound, Math.max(yTarget, lowBound));
                        }

                        updatedQValues.putScalar(0, posCurrent + transition.getAction(), yTarget);
                        posCurrent += actionCountCurrent;
                        posNext += actionCountNext;
                    }

                    return new org.nd4j.linalg.dataset.DataSet(observations, updatedQValues);
                }
            }
            : new StandardDQN(this, conf.getGamma(), conf.getErrorClamp());
    }
}
