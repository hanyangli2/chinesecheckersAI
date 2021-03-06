package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

public class DQNVarActionsFactoryStdDense implements DQNFactory {

    DQNDenseNetworkConfiguration conf;

    public DQN buildDQN(int[] numInputs, int numOutputs) {
        int nIn = 1;

        for (int i : numInputs) {
            nIn *= i;
        }

        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
            .weightInit(WeightInit.XAVIER)
            .l2(conf.getL2())
            .list()
            .layer(0,
                new DenseLayer.Builder()
                    .nIn(nIn)
                    .nOut(conf.getNumHiddenNodes())
                    .activation(Activation.RELU).build()
            );


        for (int i = 1; i < conf.getNumLayers(); i++) {
            confB.layer(i, new DenseLayer.Builder().nIn(conf.getNumHiddenNodes()).nOut(conf.getNumHiddenNodes())
                .activation(Activation.RELU).build());
        }

        confB.layer(conf.getNumLayers(),
            new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(conf.getNumHiddenNodes())
                .nOut(1)
                //.nOut(numOutputs) /// !!!
                .build()
        );


        MultiLayerConfiguration mlnconf = confB.build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        if (conf.getListeners() != null) {
            model.setListeners(conf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }
        return new DQNVarActions(model, numInputs);
    }

    public DQNVarActionsFactoryStdDense(DQNDenseNetworkConfiguration conf)
    {
        this.conf = conf;
    }

    @Deprecated
    public static class Configuration {

        int numLayer;
        int numHiddenNodes;
        double l2;
        IUpdater updater;
        TrainingListener[] listeners;

        /**
         * Converts the deprecated Configuration to the new NetworkConfiguration format
         */
        public DQNDenseNetworkConfiguration toNetworkConfiguration() {
            DQNDenseNetworkConfiguration.DQNDenseNetworkConfigurationBuilder builder = DQNDenseNetworkConfiguration.builder()
                .numHiddenNodes(numHiddenNodes)
                .numLayers(numLayer)
                .l2(l2)
                .updater(updater);

            if (listeners != null) {
                builder.listeners(Arrays.asList(listeners));
            }

            return builder.build();
        }
    }
}
